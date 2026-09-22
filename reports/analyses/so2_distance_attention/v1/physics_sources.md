# Physical distance laws and the attention diagnostic

Sources checked on 2026-09-06. These theoretical examples motivate reference
curves; they do not identify molecular transport from learned attention.

## Steady three-dimensional diffusion: concentration and flux differ

For an isotropic point source releasing Q molecules per unit time into an
unbounded homogeneous three-dimensional medium, the diffusion–degradation
solution is

`c(r) = Q exp(-r/lambda) / (4 pi D r)`, with `lambda = sqrt(D/gamma)`.

The authors give this solution and discuss regularizing the point source to
represent a finite biological source in Appendix A of
[Perez Ipiña and Camley, Competing chemical gradients change chemotactic dynamics and cell distribution](https://arxiv.org/html/2507.19341v1).
This author manuscript is associated with the published article in Physical
Review E 113, 034406 (2026).

**Derived here from that solution:** without degradation (`gamma = 0`),
concentration is proportional to `1/r`. Applying Fick's law gives the net
outward radial flux per unit area,
`J_r = -D dc/dr = Q/(4 pi r^2)`. Its integral over a sphere is Q. Thus the
inverse-square law describes this flux density under these assumptions;
it does not automatically describe concentration or neural attention.
With degradation, the same differentiation gives
`J_r = Q exp(-r/lambda) [1/r^2 + 1/(lambda r)]/(4 pi)`.

## Time dependence changes the profile

An instantaneous point release in d dimensions has concentration
`c(r,t) = N (4 pi D t)^(-d/2) exp[-r^2/(4 D t)]`.
At fixed time its distance dependence is Gaussian. Page 4 of the
[MIT 3.185 Transport Phenomena recitation notes](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/f120cb8651c167c3e102bcc3c7086748_recitation2.pdf)
lists the one-, two-, and three-dimensional kernels. The release history and
observation time therefore matter when proposing a distance law.

## Dimensionality, flow, and boundaries also matter

As a primary research example, equations 5–6 of
[Jiang and McDonald, Dissolution of plane surfaces by sources in potential flow](https://discovery.ucl.ac.uk/id/eprint/10156205/1/McDonald_1-s2.0-S0167278922002536-main.pdf)
give a two-dimensional steady advection–diffusion concentration containing
an exponential directional factor times the modified Bessel function
`K0(Pe r/2)`. A wall adds an image-source term. This is an illustration of
assumption-dependent transport profiles, not a tissue transport model.

The current diagnostic measures projected distances in a two-dimensional
tissue section. This measurement does not establish that actual transport
is confined to two dimensions. It compares observed attention with a
receiver-normalized inverse-square reference on the same graph; agreement
would establish a descriptive distance association, not diffusion, molecular
flux, predictive necessity, or biological causality.
