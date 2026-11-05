# Data generation
## accum/: Accumulation dataset
- `C{C}o{o}.py`: Addition of 2~o signed integers with digits within C
## add/: Polynomial addition dataset
- `v1d{d}C{C}.py`: Addition of a pair of univariate polynomials with degree of d and coefficients with digits within C
## charset/: Characteristic set dataset
- `v{v}t{t}e{e}C{C}L{L}.py`: A pair of polynomials with v variables, t terms, exponent numbers within e, coefficients with digits within C, and polynomials in characteristic sets with coefficients with digits within L
- `wsolve.mpl`: Maple generation of characteristic set with coefficients in integer ring#
## mul/: Polynomial multiplication dataset
- `v{v}d{d}C{C}.py`: Product of a pair of polynomials with with v variables, degree of d and coefficients with digits within C
- `v{v}t{t}E{E}C{C}.py`: Product of a pair of polynomials with v variables, t terms, exponent numbers with digits within E, coefficients with digits within C
- `v1d{d}C{C}.py`: Product of a pair of univariate polynomials with degree of d and coefficients with digits within C
- `v1d15C1.py`: Product of a pair of long polynomials, which is obtained by summing over sub-product of each pair of sub-polynomial
## prem/: Polynomial pseudodivision dataset
- `v{v}t{t}E{E}C{C}L{L}.py`: Pseudodivision of a pair of polynomials with v variables, t terms, exponent numbers with digits within E, coefficients with digits within C, and all intimediate result polynomials with coefficients with digits within L
