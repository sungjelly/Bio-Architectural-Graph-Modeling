"""Pinned MyJJu GeneMAE reproduction and deterministic campaign helpers.

The architecture in this module reproduces the task-compatible model selected
by ``cmp_20260730_myjju_genemae_10core_comparison``.  It is based on commit
``f9ef61071c7e9b2751bbd59d154c13de534e7f2f`` of the local external source.
The only architecture repair is assigning ``self.self_hidden`` before the
source-compatible self branch is constructed.

The helpers intentionally keep full-cell CP10k normalisation separate from
entry masking.  Consequently, hidden entries contribute to the library-size
denominator, as declared by the frozen exploratory contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch_geometric.nn import GATv2Conv


SOURCE_COMMIT = "f9ef61071c7e9b2751bbd59d154c13de534e7f2f"
SOURCE_FILE_SHA256: Mapping[str, str] = {
    "src/gene_mae.py": (
        "e39c3b2bbc33e4b49a9a01b1919c67e5f65919737e5f855d9970ab07b2146d5a"
    ),
    "src/graphmae.py": (
        "233e2ee1d251bb84753929be6c17b37d78b23420e943b54bb939ba5062cf8eb0"
    ),
    "scripts/gene_mae_experiment.py": (
        "4133249c1145daeed8d26f7fadeb73bbf434c15c7ad9650d338cc9f91f9f7088"
    ),
}
EXPECTED_TRAINABLE_PARAMETERS_1000 = 6_888_016


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_source(source_root: str | Path) -> dict[str, str]:
    """Verify the external source files against the frozen campaign checksums.

    Parameters
    ----------
    source_root:
        Checkout root supplied by the caller.  No external-repository location
        is hard-coded into this reusable module.

    Returns
    -------
    dict
        Observed SHA-256 digest keyed by source-relative path.

    Raises
    ------
    FileNotFoundError
        If a required source file is absent.
    RuntimeError
        If any observed digest differs from the frozen digest.
    """

    root = Path(source_root)
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for relative_path, expected in SOURCE_FILE_SHA256.items():
        path = root / relative_path
        digest = _file_sha256(path)
        observed[relative_path] = digest
        if digest != expected:
            mismatches.append(
                f"{relative_path}: expected {expected}, observed {digest}"
            )
    if mismatches:
        raise RuntimeError(
            "Pinned MyJJu source checksum mismatch: " + "; ".join(mismatches)
        )
    return observed


def drop_edge(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Randomly drop edges with the source implementation's semantics."""

    if not training or p <= 0:
        return edge_index, edge_attr
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= p
    kept_attr = edge_attr[keep] if edge_attr is not None else None
    return edge_index[:, keep], kept_attr


class ResGATEncoder(nn.Module):
    """Source-compatible residual GATv2 encoder with jumping knowledge.

    The source constructor does not override ``GATv2Conv.add_self_loops``.
    Therefore every convolution retains PyG's default internal self loops even
    though :func:`build_symmetric_knn_graph` emits no explicit self edges.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        embed: int = 128,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int = 1,
        drop_edge_p: float = 0.2,
    ) -> None:
        super().__init__()
        assert layers >= 1
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_index in range(layers):
            layer_input = in_dim if layer_index == 0 else hidden
            self.convs.append(
                GATv2Conv(
                    layer_input,
                    hidden,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden))
        self.jk = nn.Linear(hidden * layers, embed)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention: bool = False,
        use_drop_edge: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[Any]]:
        """Encode nodes using the pinned layer, residual, and dropout order."""

        edge_index_used, edge_attr_used = (
            drop_edge(
                edge_index,
                edge_attr,
                self.drop_edge_p,
                self.training,
            )
            if use_drop_edge
            else (edge_index, edge_attr)
        )
        hidden_states: list[torch.Tensor] = []
        attentions: list[Any] = []
        hidden = x
        for index, (convolution, normalisation) in enumerate(
            zip(self.convs, self.norms, strict=True)
        ):
            if return_attention:
                output, attention = convolution(
                    hidden,
                    edge_index_used,
                    edge_attr=edge_attr_used,
                    return_attention_weights=True,
                )
                attentions.append(attention)
            else:
                output = convolution(
                    hidden,
                    edge_index_used,
                    edge_attr=edge_attr_used,
                )
            output = normalisation(output)
            if index > 0:
                output = output + hidden
            output = F.elu(output)
            output = F.dropout(
                output,
                p=self.dropout,
                training=self.training,
            )
            hidden = output
            hidden_states.append(hidden)
        embedding = self.jk(torch.cat(hidden_states, dim=1))
        if return_attention:
            return embedding, attentions
        return embedding


def sce_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return the source-compatible row-wise scaled cosine error."""

    x_normalised = F.normalize(x, p=2, dim=-1)
    reconstruction_normalised = F.normalize(reconstruction, p=2, dim=-1)
    cosine = (x_normalised * reconstruction_normalised).sum(-1)
    return ((1 - cosine).clamp_min(0) ** gamma).mean()


class GeneMAE(nn.Module):
    """Pinned feature-masking GeneMAE with the declared self-branch repair."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        embed: int = 128,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = 1,
        drop_edge_p: float = 0.2,
        mask_rate: float = 0.5,
        huber_delta: float = 1.0,
        sce_weight: float = 0.0,
        gamma: float = 2.0,
        gnn_decoder: bool = True,
        self_branch: bool = True,
        self_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.mask_rate = mask_rate
        self.huber_delta = huber_delta
        self.sce_weight = sce_weight
        self.gamma = gamma
        self.self_branch = self_branch
        # The external source reads this attribute without assigning it.  This
        # is the sole source repair authorised by the frozen campaign contract.
        self.self_hidden = int(self_hidden)
        self.encoder = ResGATEncoder(
            in_dim,
            hidden,
            embed,
            heads,
            layers,
            dropout,
            edge_dim,
            drop_edge_p,
        )

        decoder_input = embed
        if self_branch:
            self.self_enc = nn.Sequential(
                nn.Linear(in_dim, self.self_hidden),
                nn.LayerNorm(self.self_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.self_hidden, self.self_hidden),
                nn.LayerNorm(self.self_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.self_hidden, embed),
            )
            decoder_input = 2 * embed
        self.dec_proj = nn.Linear(decoder_input, hidden)
        if gnn_decoder:
            self.decoder = GATv2Conv(
                hidden,
                in_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
            )
            self.gnn_decoder = True
        else:
            self.decoder = nn.Sequential(
                nn.ELU(),
                nn.Linear(hidden, in_dim),
            )
            self.gnn_decoder = False
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

    def _mask(
        self,
        x: torch.Tensor,
        entry_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if entry_mask is not None:
            return entry_mask
        random_values = torch.rand(
            x.shape,
            device=x.device,
            generator=generator,
        )
        mask = random_values < self.mask_rate
        if mask.sum() == 0:
            mask.view(-1)[0] = True
        return mask

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._mask(x, entry_mask, generator)
        masked_input = torch.where(
            mask,
            self.mask_token.expand_as(x),
            x,
        )
        embedding = self.encoder(
            masked_input,
            edge_index,
            edge_attr=edge_attr,
        )
        if self.self_branch:
            embedding = torch.cat(
                [embedding, self.self_enc(masked_input)],
                dim=-1,
            )
        hidden = F.elu(self.dec_proj(embedding))
        reconstruction = (
            self.decoder(hidden, edge_index, edge_attr=edge_attr)
            if self.gnn_decoder
            else self.decoder(hidden)
        )
        return reconstruction, mask

    def loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        reconstruction, mask = self.forward(
            x,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
            generator=generator,
        )
        regression = F.huber_loss(
            reconstruction[mask],
            x[mask],
            delta=self.huber_delta,
        )
        if self.sce_weight > 0:
            regression = regression + self.sce_weight * sce_loss(
                x,
                reconstruction,
                self.gamma,
            )
        return regression

    @torch.no_grad()
    def recon_eval(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        return self.forward(
            x,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
            generator=generator,
        )

    @torch.no_grad()
    def embed(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        encoded = self.encoder(
            x,
            edge_index,
            edge_attr=edge_attr,
            use_drop_edge=False,
        )
        assert isinstance(encoded, torch.Tensor)
        return encoded

    @torch.no_grad()
    def attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> list[Any]:
        self.eval()
        _, attentions = self.encoder(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention=True,
            use_drop_edge=False,
        )
        return attentions


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters with the external source's definition."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def make_source_model(
    num_genes: int = 1000,
    mask_rate: float = 0.5,
    **architecture_overrides: Any,
) -> GeneMAE:
    """Construct the source report's selected dual-path MLP-decoder model."""

    configuration: dict[str, Any] = {
        "hidden": 256,
        "embed": 192,
        "heads": 6,
        "layers": 4,
        "dropout": 0.2,
        "edge_dim": 1,
        "drop_edge_p": 0.2,
        "huber_delta": 1.0,
        "sce_weight": 0.0,
        "gnn_decoder": False,
        "self_branch": True,
        "self_hidden": 512,
    }
    configuration.update(architecture_overrides)
    return GeneMAE(
        in_dim=int(num_genes),
        mask_rate=float(mask_rate),
        **configuration,
    )


def log1p_cp10k(
    counts: np.ndarray | torch.Tensor,
    target_sum: float = 10_000.0,
) -> np.ndarray | torch.Tensor:
    """Apply full-cell CP10k normalisation followed by ``log1p``.

    The returned container matches the input container.  Values are float32,
    matching the external data path and model input.  Rows with zero library
    size remain all zero.
    """

    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and strictly positive.")
    if isinstance(counts, torch.Tensor):
        if counts.ndim != 2:
            raise ValueError("counts must have shape [cells, genes].")
        values = counts.to(dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("counts must contain only finite values.")
        if bool((values < 0).any()):
            raise ValueError("counts must be nonnegative.")
        library_size = values.sum(dim=1, keepdim=True)
        safe_library_size = torch.where(
            library_size == 0,
            torch.ones_like(library_size),
            library_size,
        )
        return torch.log1p(values / safe_library_size * float(target_sum))

    values_array = np.asarray(counts)
    if values_array.ndim != 2:
        raise ValueError("counts must have shape [cells, genes].")
    values_float = values_array.astype(np.float32, copy=False)
    if not np.isfinite(values_float).all():
        raise ValueError("counts must contain only finite values.")
    if np.any(values_float < 0):
        raise ValueError("counts must be nonnegative.")
    library_size_array = values_float.sum(
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    safe_library_size_array = np.where(
        library_size_array == 0,
        np.float32(1.0),
        library_size_array,
    )
    normalised = (
        values_float / safe_library_size_array * np.float32(target_sum)
    )
    return np.log1p(normalised).astype(np.float32, copy=False)


def recursive_spatial_tiles(
    coords: np.ndarray,
    max_nodes: int = 7000,
) -> tuple[np.ndarray, ...]:
    """Deterministically split coordinates along median longest-axis cuts."""

    coordinates = np.asarray(coords)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coords must have shape [nodes, 2].")
    if not np.isfinite(coordinates).all():
        raise ValueError("coords must contain only finite values.")
    if isinstance(max_nodes, bool) or int(max_nodes) != max_nodes or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer.")
    cap = int(max_nodes)
    if coordinates.shape[0] == 0:
        return ()

    def split(indices: np.ndarray) -> list[np.ndarray]:
        if indices.size <= cap:
            return [indices]
        local_coordinates = coordinates[indices]
        ranges = np.ptp(local_coordinates, axis=0)
        axis = 0 if ranges[0] >= ranges[1] else 1
        median = np.median(local_coordinates[:, axis])
        left = indices[local_coordinates[:, axis] <= median]
        right = indices[local_coordinates[:, axis] > median]
        if left.size == 0 or right.size == 0:
            order = np.argsort(
                local_coordinates[:, axis],
                kind="stable",
            )
            ordered_indices = indices[order]
            halfway = ordered_indices.size // 2
            left = ordered_indices[:halfway]
            right = ordered_indices[halfway:]
        return split(left) + split(right)

    initial = np.arange(coordinates.shape[0], dtype=np.int64)
    return tuple(split(initial))


def gaussian_edge_features(
    coords: np.ndarray,
    edge_index: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute the pinned one-channel Gaussian distance edge feature."""

    coordinates = np.asarray(coords)
    edges = np.asarray(edge_index)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coords must have shape [nodes, 2].")
    if not np.isfinite(coordinates).all():
        raise ValueError("coords must contain only finite values.")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges].")
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("edge_index must use an integer dtype.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and strictly positive.")
    if edges.shape[1] == 0:
        return np.empty((0, 1), dtype=np.float32)
    if edges.min() < 0 or edges.max() >= coordinates.shape[0]:
        raise ValueError("edge_index contains an out-of-range node index.")

    source, target = edges
    distances = np.linalg.norm(
        coordinates[source] - coordinates[target],
        axis=1,
    )
    sigma = float(scale) * (float(np.median(distances)) + 1e-8)
    features = np.exp(
        -(distances**2) / (2.0 * sigma**2),
    )
    return features.astype(np.float32, copy=False)[:, None]


def _deterministic_neighbours(
    coordinates: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return k neighbours per row, resolving distance ties by node index."""

    node_count = coordinates.shape[0]
    if node_count <= 1:
        return np.empty((node_count, 0), dtype=np.int64)
    neighbour_count = min(k, node_count - 1)
    tree = cKDTree(coordinates)
    neighbours = np.empty(
        (node_count, neighbour_count),
        dtype=np.int64,
    )
    if neighbour_count == node_count - 1:
        all_indices = np.arange(node_count, dtype=np.int64)
        for node in range(node_count):
            candidates = all_indices[all_indices != node]
            squared_distances = np.sum(
                (coordinates[candidates] - coordinates[node]) ** 2,
                axis=1,
            )
            order = np.lexsort((candidates, squared_distances))
            neighbours[node] = candidates[order]
        return neighbours

    for node in range(node_count):
        query_distances, query_indices = tree.query(
            coordinates[node],
            k=neighbour_count + 1,
        )
        query_distances = np.atleast_1d(query_distances)
        query_indices = np.atleast_1d(query_indices).astype(
            np.int64,
            copy=False,
        )
        nonself_query = query_indices[query_indices != node]
        if nonself_query.size < neighbour_count:
            raise RuntimeError("kNN query did not return enough non-self nodes.")
        boundary_distance = float(query_distances[-1])
        radius = np.nextafter(boundary_distance, np.inf)
        candidates = np.asarray(
            tree.query_ball_point(coordinates[node], radius),
            dtype=np.int64,
        )
        candidates = candidates[candidates != node]
        squared_distances = np.sum(
            (coordinates[candidates] - coordinates[node]) ** 2,
            axis=1,
        )
        order = np.lexsort((candidates, squared_distances))
        ordered_candidates = candidates[order]
        if ordered_candidates.size < neighbour_count:
            raise RuntimeError("kNN radius query returned too few nodes.")
        neighbours[node] = ordered_candidates[:neighbour_count]
    return neighbours


def build_symmetric_knn_graph(
    coords: np.ndarray,
    k: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic symmetric-union spatial kNN graph.

    Returns an integer ``edge_index`` with shape ``[2, E]`` and the pinned
    Gaussian edge feature with shape ``[E, 1]``.  Both directions of every
    union edge are present and self edges are absent.
    """

    coordinates = np.asarray(coords)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coords must have shape [nodes, 2].")
    if not np.isfinite(coordinates).all():
        raise ValueError("coords must contain only finite values.")
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer.")
    node_count = coordinates.shape[0]
    if node_count <= 1:
        edges = np.empty((2, 0), dtype=np.int64)
        return edges, np.empty((0, 1), dtype=np.float32)

    neighbours = _deterministic_neighbours(coordinates, int(k))
    sources = np.repeat(
        np.arange(node_count, dtype=np.int64),
        neighbours.shape[1],
    )
    targets = neighbours.reshape(-1)
    directed_pairs = np.column_stack([sources, targets])
    reverse_pairs = directed_pairs[:, ::-1]
    symmetric_pairs = np.unique(
        np.concatenate([directed_pairs, reverse_pairs], axis=0),
        axis=0,
    )
    edge_index = symmetric_pairs.T.astype(np.int64, copy=False)
    edge_attr = gaussian_edge_features(coordinates, edge_index)
    return edge_index, edge_attr


def permute_graph_node_labels(
    edge_index: np.ndarray | torch.Tensor,
    num_nodes: int,
    seed: int,
) -> np.ndarray | torch.Tensor:
    """Relabel both edge rows with one deterministic node permutation.

    Relabelling preserves graph topology, edge count, and the degree sequence,
    while breaking the assignment between graph positions and unchanged cell
    feature rows.  The output container, dtype, and torch device match the
    input.
    """

    if isinstance(num_nodes, bool) or int(num_nodes) != num_nodes or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer.")
    node_count = int(num_nodes)
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer.")

    is_tensor = isinstance(edge_index, torch.Tensor)
    edges_array = (
        edge_index.detach().cpu().numpy()
        if is_tensor
        else np.asarray(edge_index)
    )
    if edges_array.ndim != 2 or edges_array.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges].")
    if not np.issubdtype(edges_array.dtype, np.integer):
        raise TypeError("edge_index must use an integer dtype.")
    if edges_array.size and (
        edges_array.min() < 0 or edges_array.max() >= node_count
    ):
        raise ValueError("edge_index contains an out-of-range node index.")

    permutation = np.random.default_rng(int(seed)).permutation(node_count)
    relabelled = permutation[edges_array].astype(
        edges_array.dtype,
        copy=False,
    )
    if is_tensor:
        return torch.as_tensor(
            relabelled,
            dtype=edge_index.dtype,
            device=edge_index.device,
        )
    return relabelled


def sample_entry_mask(
    shape: Sequence[int],
    rate: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample a reproducible Bernoulli entry mask using a CPU RNG stream."""

    dimensions = tuple(int(dimension) for dimension in shape)
    if len(dimensions) == 0 or any(dimension < 0 for dimension in dimensions):
        raise ValueError("shape must contain nonnegative dimensions.")
    if int(np.prod(dimensions, dtype=np.int64)) == 0:
        raise ValueError("shape must contain at least one entry.")
    if not np.isfinite(rate) or rate < 0 or rate > 1:
        raise ValueError("rate must lie in [0, 1].")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    mask = torch.rand(dimensions, generator=generator) < float(rate)
    if mask.sum() == 0:
        mask.view(-1)[0] = True
    return mask.to(device=device)


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class MaskedRegressionAccumulator:
    """Streaming sufficient statistics for the campaign's common metrics."""

    def __init__(
        self,
        num_genes: int,
        *,
        huber_delta: float = 1.0,
        variance_epsilon: float = 1e-12,
    ) -> None:
        if (
            isinstance(num_genes, bool)
            or int(num_genes) != num_genes
            or num_genes < 1
        ):
            raise ValueError("num_genes must be a positive integer.")
        if not np.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError("huber_delta must be finite and strictly positive.")
        if not np.isfinite(variance_epsilon) or variance_epsilon < 0:
            raise ValueError("variance_epsilon must be finite and nonnegative.")

        self.num_genes = int(num_genes)
        self.huber_delta = float(huber_delta)
        self.variance_epsilon = float(variance_epsilon)
        self.n_masked = 0
        self.sum_true = 0.0
        self.sum_prediction = 0.0
        self.sum_true_square = 0.0
        self.sum_prediction_square = 0.0
        self.sum_cross = 0.0
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_huber = 0.0
        self.gene_n = np.zeros(self.num_genes, dtype=np.int64)
        self.gene_sum_true = np.zeros(self.num_genes, dtype=np.float64)
        self.gene_sum_prediction = np.zeros(
            self.num_genes,
            dtype=np.float64,
        )
        self.gene_sum_true_square = np.zeros(
            self.num_genes,
            dtype=np.float64,
        )
        self.gene_sum_prediction_square = np.zeros(
            self.num_genes,
            dtype=np.float64,
        )
        self.gene_sum_cross = np.zeros(
            self.num_genes,
            dtype=np.float64,
        )
        self.cell_pearson_sum = 0.0
        self.cell_pearson_count = 0

    def update(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
    ) -> None:
        """Update statistics from one row-aligned cell chunk."""

        true = _as_numpy(y_true)
        prediction = _as_numpy(y_pred)
        mask_array = _as_numpy(mask)
        expected_shape = (true.shape[0], self.num_genes) if true.ndim == 2 else None
        if expected_shape is None or true.shape != expected_shape:
            raise ValueError(
                f"y_true must have shape [cells, {self.num_genes}]."
            )
        if prediction.shape != true.shape or mask_array.shape != true.shape:
            raise ValueError("y_true, y_pred, and mask must have identical shapes.")
        if not np.issubdtype(mask_array.dtype, np.bool_):
            raise TypeError("mask must use a boolean dtype.")

        true_float = true.astype(np.float64, copy=False)
        prediction_float = prediction.astype(np.float64, copy=False)
        if not np.isfinite(true_float[mask_array]).all():
            raise ValueError("Masked target values must be finite.")
        if not np.isfinite(prediction_float[mask_array]).all():
            raise ValueError("Masked predictions must be finite.")

        selected_true = true_float[mask_array]
        selected_prediction = prediction_float[mask_array]
        selected_count = int(selected_true.size)
        if selected_count == 0:
            return
        error = selected_prediction - selected_true
        absolute_error = np.abs(error)
        delta = self.huber_delta

        self.n_masked += selected_count
        self.sum_true += float(selected_true.sum(dtype=np.float64))
        self.sum_prediction += float(
            selected_prediction.sum(dtype=np.float64)
        )
        self.sum_true_square += float(
            np.square(selected_true).sum(dtype=np.float64)
        )
        self.sum_prediction_square += float(
            np.square(selected_prediction).sum(dtype=np.float64)
        )
        self.sum_cross += float(
            (selected_true * selected_prediction).sum(dtype=np.float64)
        )
        self.sum_squared_error += float(
            np.square(error).sum(dtype=np.float64)
        )
        self.sum_absolute_error += float(
            absolute_error.sum(dtype=np.float64)
        )
        huber = np.where(
            absolute_error <= delta,
            0.5 * np.square(error),
            delta * (absolute_error - 0.5 * delta),
        )
        self.sum_huber += float(huber.sum(dtype=np.float64))

        masked_true = np.where(mask_array, true_float, 0.0)
        masked_prediction = np.where(mask_array, prediction_float, 0.0)
        self.gene_n += mask_array.sum(axis=0, dtype=np.int64)
        self.gene_sum_true += masked_true.sum(axis=0, dtype=np.float64)
        self.gene_sum_prediction += masked_prediction.sum(
            axis=0,
            dtype=np.float64,
        )
        self.gene_sum_true_square += np.square(masked_true).sum(
            axis=0,
            dtype=np.float64,
        )
        self.gene_sum_prediction_square += np.square(
            masked_prediction
        ).sum(axis=0, dtype=np.float64)
        self.gene_sum_cross += (
            masked_true * masked_prediction
        ).sum(axis=0, dtype=np.float64)

        cell_n = mask_array.sum(axis=1, dtype=np.int64)
        cell_sum_true = masked_true.sum(axis=1, dtype=np.float64)
        cell_sum_prediction = masked_prediction.sum(
            axis=1,
            dtype=np.float64,
        )
        cell_sum_true_square = np.square(masked_true).sum(
            axis=1,
            dtype=np.float64,
        )
        cell_sum_prediction_square = np.square(
            masked_prediction
        ).sum(axis=1, dtype=np.float64)
        cell_sum_cross = (
            masked_true * masked_prediction
        ).sum(axis=1, dtype=np.float64)
        valid_count = cell_n >= 2
        safe_n = np.where(valid_count, cell_n, 1).astype(np.float64)
        cell_true_ss = (
            cell_sum_true_square
            - np.square(cell_sum_true) / safe_n
        )
        cell_prediction_ss = (
            cell_sum_prediction_square
            - np.square(cell_sum_prediction) / safe_n
        )
        cell_cross_ss = (
            cell_sum_cross
            - cell_sum_true * cell_sum_prediction / safe_n
        )
        valid_cell = (
            valid_count
            & (cell_true_ss > self.variance_epsilon)
            & (cell_prediction_ss > self.variance_epsilon)
        )
        if np.any(valid_cell):
            correlations = cell_cross_ss[valid_cell] / np.sqrt(
                cell_true_ss[valid_cell]
                * cell_prediction_ss[valid_cell]
            )
            self.cell_pearson_sum += float(
                correlations.sum(dtype=np.float64)
            )
            self.cell_pearson_count += int(correlations.size)

    @staticmethod
    def _correlation(
        count: np.ndarray | float | int,
        sum_x: np.ndarray | float,
        sum_y: np.ndarray | float,
        sum_x_square: np.ndarray | float,
        sum_y_square: np.ndarray | float,
        sum_cross: np.ndarray | float,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        count_array = np.asarray(count, dtype=np.float64)
        safe_count = np.where(count_array > 0, count_array, 1.0)
        x_ss = np.asarray(sum_x_square) - np.square(sum_x) / safe_count
        y_ss = np.asarray(sum_y_square) - np.square(sum_y) / safe_count
        cross_ss = np.asarray(sum_cross) - np.asarray(sum_x) * np.asarray(
            sum_y
        ) / safe_count
        valid = (count_array >= 2) & (x_ss > epsilon) & (y_ss > epsilon)
        correlation = np.full(count_array.shape, np.nan, dtype=np.float64)
        correlation[valid] = cross_ss[valid] / np.sqrt(
            x_ss[valid] * y_ss[valid]
        )
        return correlation, valid

    def per_gene_pearson(self) -> np.ndarray:
        """Return one Pearson coefficient per gene, with undefined values NaN."""

        correlation, _ = self._correlation(
            self.gene_n,
            self.gene_sum_true,
            self.gene_sum_prediction,
            self.gene_sum_true_square,
            self.gene_sum_prediction_square,
            self.gene_sum_cross,
            self.variance_epsilon,
        )
        return correlation

    def finalize(self) -> dict[str, float | int]:
        """Return scalar campaign metrics from all accumulated chunks."""

        if self.n_masked == 0:
            raise ValueError("Cannot finalize metrics without masked entries.")
        pooled_array, pooled_valid = self._correlation(
            self.n_masked,
            self.sum_true,
            self.sum_prediction,
            self.sum_true_square,
            self.sum_prediction_square,
            self.sum_cross,
            self.variance_epsilon,
        )
        pooled_pearson = (
            float(pooled_array)
            if bool(pooled_valid)
            else float("nan")
        )
        target_ss = (
            self.sum_true_square
            - self.sum_true**2 / float(self.n_masked)
        )
        masked_r2 = (
            1.0 - self.sum_squared_error / target_ss
            if target_ss > self.variance_epsilon
            else float("nan")
        )
        gene_pearson = self.per_gene_pearson()
        finite_gene = np.isfinite(gene_pearson)
        gene_mean = (
            float(gene_pearson[finite_gene].mean())
            if np.any(finite_gene)
            else float("nan")
        )
        gene_median = (
            float(np.median(gene_pearson[finite_gene]))
            if np.any(finite_gene)
            else float("nan")
        )
        cell_mean = (
            self.cell_pearson_sum / self.cell_pearson_count
            if self.cell_pearson_count > 0
            else float("nan")
        )
        return {
            "n_masked": self.n_masked,
            "masked_huber": self.sum_huber / self.n_masked,
            "masked_mse": self.sum_squared_error / self.n_masked,
            "masked_mae": self.sum_absolute_error / self.n_masked,
            "pooled_pearson": pooled_pearson,
            "masked_r2": masked_r2,
            "gene_pearson_mean": gene_mean,
            "gene_pearson_median": gene_median,
            "cell_pearson_mean": cell_mean,
            "n_valid_genes": int(finite_gene.sum()),
            "n_valid_cells": self.cell_pearson_count,
        }


def masked_regression_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    huber_delta: float = 1.0,
) -> dict[str, float | int]:
    """Compute the common masked metrics through the streaming accumulator."""

    true = _as_numpy(y_true)
    if true.ndim != 2:
        raise ValueError("y_true must have shape [cells, genes].")
    accumulator = MaskedRegressionAccumulator(
        true.shape[1],
        huber_delta=huber_delta,
    )
    accumulator.update(y_true, y_pred, mask)
    return accumulator.finalize()


__all__ = [
    "EXPECTED_TRAINABLE_PARAMETERS_1000",
    "GeneMAE",
    "MaskedRegressionAccumulator",
    "ResGATEncoder",
    "SOURCE_COMMIT",
    "SOURCE_FILE_SHA256",
    "build_symmetric_knn_graph",
    "count_parameters",
    "drop_edge",
    "gaussian_edge_features",
    "log1p_cp10k",
    "make_source_model",
    "masked_regression_metrics",
    "permute_graph_node_labels",
    "sample_entry_mask",
    "sce_loss",
    "recursive_spatial_tiles",
    "verify_pinned_source",
]
