# weight initialization
## why do we have to initialize
**1. when initializing all weights to zero** <br/>
&emsp;1) if Activation function is **ReLU** <br/>
&emsp;&emsp; then every outputs of first layer would be zero <br/>
&emsp;&emsp; therefore, dying ReLU occurs  <br/>
&emsp;2) if Activation function is **sigmoid or tanh** <br/>
&emsp;&emsp; every outputs are 0.5 or 0 <br/>
&emsp;&emsp; while gradient descent <br/>
&emsp;&emsp;&emsp; $\frac {\partial J} {\partial O_i} \frac {\partial O_i} {\partial O_{i-1}} \frac {\partial O_{i-1}} {\partial O_{i-2}}...$ <br/>
&emsp;&emsp;&emsp; $\frac {\partial O_i} {\partial O_{i-1}} = \frac {\partial f(wx+b)} {\partial x} = 0 (\because w = 0)$ <br/>
&emsp;&emsp;&emsp; therefore, no updates

**2. when every weights has same value(except 0)** <br/>
&emsp; &bull; updated as same value <br/>
&emsp; &bull; Hidden layer losts its purpose<br/>
&emsp;&emsp; <img width="333" alt="image" src="https://github.com/user-attachments/assets/2075042b-ee8d-413b-bcd9-94d860ab1de9">
 <br/> <br/>
**therefore, we need to initialize weights appropriately**

**2. initialize with normail distribution** <br/>
&emsp; **case 1.** $w\sim N(0,1)$ which is large variance <br/>
&emsp;&emsp; <img width="333" alt="image" src="https://github.com/user-attachments/assets/0bdb28fe-54e5-4a9a-9dda-5271ea5c3558"> <br/>
&emsp;&emsp; &bull; this prcture is distribution of weights for layers <br/>
&emsp;&emsp; &bull; every layers uses sigmoid function for activation <br/>
&emsp;&emsp; &bull; Most weights are biased toward 0 and 1. Then it will occur gradient vanishing while gradient descent <br/>
&emsp; **case 2.** $w\sim N(0,0.01)$ which is small variance <br/>
&emsp;&emsp; <img width="333" alt="image" src="https://github.com/user-attachments/assets/7303b2d1-ecd9-40d5-bffb-2ffe17670aa6"> <br/>
&emsp;&emsp; &bull; becomes similar to having all weights with the same value  <br/>
&emsp;&emsp; &bull; therefore, Hidden layer losts its purpose <br/> <br/>
**therefore, we should use appropriate variance**

**3. Xavier initialization** <br/>
&emsp; initialize with optimize variance according to number of input and output nodes <br/>
&emsp; Normal : $w \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$ <br/>
&emsp; Uniform : $w \sim N\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$ <br/>
&emsp;&emsp; <img width="333" alt="image" src="https://github.com/user-attachments/assets/f39194df-0b38-4dbe-836b-11aa5f179edf"> <br/>
&emsp; **therfore, shows good performance when combined with sigmoid or tanh functions**<br/>
<br/>&emsp;**why it fits well with logistc functions?** <br/>
The densely packed values from a Gaussian distribution align with the approximately linear part of the logistic function.
 <br/> <br/>
&emsp; **But when it combined with ReLU** <br/>
&emsp;&emsp; <img width="333" alt="image" src="https://github.com/user-attachments/assets/0435f563-eda6-4ac5-89dd-913e030a5fe4"> <br/>
&emsp; occurs dying ReLU

**4. He initializaiton** <br/>
&emsp; To, decrease the problem of Xavier initialization with **ReLU** <br/>
&emsp; Normal : $w \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$ <br/>
&emsp; Uniform : $w \sim N\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$ <br/>
&emsp;&emsp;<img width="371" alt="image" src="https://github.com/user-attachments/assets/c08f8c4f-8495-4f6c-9cb4-08949545503f"> <br/>


