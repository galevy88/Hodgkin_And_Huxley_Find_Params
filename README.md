# Hodjkin_And_Huxley_Find_Params
## Generating Data Base On Equations

$$\alpha = \frac{0.01(V+55)}{1-e^{-(\frac{V+55}{10})}}$$
$$\hspace{1cm}$$
$$\beta = 0.125e^{-(\frac{V+65}{80})}$$
$$\hspace{1cm}$$
$$\tau = \frac{1}{\alpha + \beta}$$
$$\hspace{1cm}$$
$$n_\infty = \frac{\alpha}{\alpha + \beta}$$
$$\hspace{1cm}$$
$$n = n_\infty(1-e^{\frac{-t}{\tau}})$$
$$I_k = \hat{g}_k * n^4(1-e^{\frac{-t}{\tau}})(V-E_k)$$
