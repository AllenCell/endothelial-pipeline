Distilled and verified: 82 tests pass, workflow runs clean.

## What the implementation is now

[noise_signature.py](src/endo_pipeline/library/analyze/numerics/noise_signature.py) — five things, one entry point `compute_noise_signature`:

| function | purpose |
|---|---|
| `compute_lagged_covariances` / `normalize_lagged_covariances` | residual ACF $\rho_{ij}(\tau)$, pairwise-complete across gaps |
| `compute_integrated_correlation_time` | $\tau_c$, initial-positive-sequence truncation |
| `compute_spectral_exponent` (+ `compute_power_spectrum`) | $\beta$ from $S(f)\sim f^{-\beta}$ |
| `compute_measurement_noise_variance` / `_fraction` | $\sigma_\epsilon^2$ from $\rho(1)$ |
| `compute_reliability_ratio` | drift attenuation $\lambda$ |

`permute_residuals_in_time` calibrates all of them. Removed: Ljung–Box, Hosking, Bartlett KS, the MSD variance-scaling route, the diffusion-coefficient statistics, the corrected-amplitude plot, and the now-empty `p_value_analytic` column. The `time_lag` fix on `get_trajectories_and_differences_for_noise_correlations` stays — it repairs a real `TypeError` in `compute_msd`.

## Results

| | $\rho(1)$ | $\tau_c$ | $\beta$ | $\sigma_\epsilon^2$ | meas. frac. | $\lambda$ |
|---|---|---|---|---|---|---|
| $\theta$ | −0.274 | 0.500 | −0.573 | 7.5e-4 | 0.548 | 0.9991 |
| $r$ | −0.268 | 0.500 | −0.498 | 1.80e-3 | 0.535 | 0.9917 |
| PC3 | −0.258 | 0.500 | −0.430 | 1.70e-3 | 0.516 | 0.9923 |

$\rho(1)$ and $\beta$ reject at the permutation floor ($p = 0.005$, $|z| = 16$–$39$); $\tau_c$ does not reject at all ($p = 0.70$–$0.93$).

---

# Can these series be modeled as the solution of a Markov SDE?

**Yes — provided the model includes an observation-noise layer.** The data are consistent with a Markov SDE observed with independent error, and inconsistent with a non-Markovian or colored-noise process.

### The residuals are not white

$\rho(1) \approx -0.26$ at $z = -19$ to $-39$. Under the Euler–Maruyama discretization $\Delta x = f(x)\Delta t + \sigma\,\Delta W$, residuals must be independent increments, so this rejects the naive model outright.

### But they fail in the wrong direction for colored noise

Memory in the driving noise means $\rho(1) > 0$ and $\beta > 0$. Observed: $\rho(1) < 0$ and $\beta \approx -0.43$ to $-0.57$ — power *rising* with frequency. The departure is anti-persistent, which is the signature of differencing, not of memory (Box, Jenkins & Reinsel 2008, Ch. 3; Priestley 1981, §6.2). An Ornstein–Uhlenbeck or $1/f$ driving term is excluded.

### The decorrelation time is exactly that of white noise

$\tau_c = 0.5000$ with $p = 0.70$–$0.93$, matching the white-noise value. Estimated over the initial positive sequence (Geyer 1992), $\tau_c$ sums *all* lags with persistent memory; here it truncates at lag 1 because $\rho(1)<0$, and nothing accumulates beyond. **Large $|\rho(1)|$ with white $\tau_c$ means the entire violation occupies a single lag** — the defining property of an MA(1) process. Every form of extended memory is ruled out.

### MA(1) is exactly what measurement error produces

Observing $x$ with independent error $\epsilon$ gives $\Delta x(t) = s(t) + \epsilon(t{+}1) - \epsilon(t)$. The shared term $\epsilon(t{+}1)$ enters consecutive residuals with opposite signs, forcing $\mathrm{Cov}(\eta_t,\eta_{t+1}) = -\sigma_\epsilon^2$ and zero beyond — no free parameters. This is the structure behind the covariance-based estimator of Vestergaard, Blainey & Flyvbjerg (2014), and the reason localization error mimics anomalous diffusion in tracking data (Martin, Forstner & Käs 2002; Savin & Doyle 2005; Michalet 2010; Berglund 2010). Inverting it gives $\sigma_\epsilon^2 = -\rho(1)\mathrm{Var}(\eta)$: about **52% of residual variance is observation error, not dynamics**, consistently across all three features.

### Why this does not break Markovianity

Measurement error renders the *observed* sequence non-Markov while the *latent* state stays Markov — a hidden Markov / state-space model, not a non-Markovian process. The correct model class is therefore

$$dx = f(x)\,dt + \sigma\,dW, \qquad y_k = x(t_k) + \epsilon_k,\ \ \epsilon_k \sim \mathcal{N}(0,\sigma_\epsilon^2)$$

This is standard in drift–diffusion reconstruction from noisy data (Böttcher et al. 2006; Lehle 2011; Friedrich, Peinke, Sahimi & Tabar 2011, §4). Note that these tests establish *consistency* with Markov dynamics, not proof — direct confirmation needs a Chapman–Kolmogorov or conditional-independence test on $x$ (Friedrich et al. 2011).

### Practical consequence: diffusion is biased, drift is not

$\mathrm{Var}(\eta)$ overstates $2D\Delta t$ by $2\sigma_\epsilon^2$, so **$D$ is inflated ~2×**. But $\lambda = 0.992$–$0.999$: drift attenuation is **under 1%**, so the vector field, fixed points, and flow field are sound. The asymmetry is just signal-to-noise — diffusion is read from differences of adjacent points (SNR $\approx 1$), drift from absolute position against a population spread 12–31× larger than $\sigma_\epsilon$. This is classical regression dilution (Spearman 1904; Fuller 1987; Carroll et al. 2006; Frost & Thompson 2000).

### References

- Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (2008). *Time Series Analysis: Forecasting and Control*, 4th ed. Wiley.
- Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
- Welch, P.D. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Trans. Audio Electroacoust.* 15(2), 70–73.
- Geyer, C.J. (1992). Practical Markov chain Monte Carlo. *Statistical Science* 7(4), 473–483.
- Theiler, J. et al. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D* 58, 77–94.
- Schreiber, T. & Schmitz, A. (2000). Surrogate time series. *Physica D* 142, 346–382.
- Vestergaard, C.L., Blainey, P.C. & Flyvbjerg, H. (2014). Optimal estimation of diffusion coefficients from single-particle trajectories. *Phys. Rev. E* 89, 022726.
- Michalet, X. (2010). MSD analysis of single-particle trajectories with localization error. *Phys. Rev. E* 82, 041914 (Erratum: 83, 059904, 2011).
- Berglund, A.J. (2010). Statistics of camera-based single-particle tracking. *Phys. Rev. E* 82, 011917.
- Savin, T. & Doyle, P.S. (2005). Static and dynamic errors in particle tracking microrheology. *Biophys. J.* 88, 623–638.
- Martin, D.S., Forstner, M.B. & Käs, J.A. (2002). Apparent subdiffusion inherent to single particle tracking. *Biophys. J.* 83, 2109–2117.
- Spearman, C. (1904). The proof and measurement of association between two things. *Am. J. Psychol.* 15, 72–101.
- Fuller, W.A. (1987). *Measurement Error Models*. Wiley.
- Carroll, R.J. et al. (2006). *Measurement Error in Nonlinear Models*, 2nd ed. Chapman & Hall/CRC.
- Frost, C. & Thompson, S.G. (2000). Correcting for regression dilution bias. *JRSS A* 163, 173–189.
- Friedrich, R., Peinke, J., Sahimi, M. & Tabar, M.R.R. (2011). Approaching complexity by stochastic methods. *Physics Reports* 506, 87–162.
- Böttcher, F. et al. (2006). Reconstruction of complex dynamical systems affected by strong measurement noise. *Phys. Rev. Lett.* 97, 090603.
- Lehle, B. (2011). Analysis of stochastic time series in the presence of strong measurement noise. *Phys. Rev. E* 83, 021113.
