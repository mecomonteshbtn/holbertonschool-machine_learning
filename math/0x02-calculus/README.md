# 0x02. Calculus

---
## Resources

## Read or watch:

    [Sigma Notation (starting at 0:32)](https://www.youtube.com/watch?v=TjMLzklnn2c)
    [Π Product Notation (up to 0:20)](https://www.youtube.com/watch?v=sP1-EQJKSgk)
    [Sigma and Pi Notation](https://mathmaine.com/2010/04/01/sigma-and-pi-notation/)
    [What is a Series?](https://virtualnerd.com/algebra-2/sequences-series/define/defining-series/series-definition)
    [What is a Mathematical Series?](https://www.quickanddirtytips.com/education/math/what-is-a-mathematical-series)
    [List of mathematical series: Sums of powers](https://en.wikipedia.org/wiki/List_of_mathematical_series#Sums_of_powers)
    [Bernoulli Numbers(Bn)](https://en.wikipedia.org/wiki/Bernoulli_number)
    [Bernoulli Polynomials(Bn(x))](https://en.wikipedia.org/wiki/Bernoulli_polynomials)
    [Derivative (mathematics)](https://simple.wikipedia.org/wiki/Derivative_%28mathematics%29)
    [Calculus for ML](https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html)
    [1 of 2: Seeing the big picture](https://www.youtube.com/watch?v=tt2DGYOi3hc)
    [2 of 2: First Principles](https://www.youtube.com/watch?v=50Bda5VKbqA)
    [1 of 2: Finding the Derivative](https://www.youtube.com/watch?v=fXYhyyJpFe8)
    [2 of 2: What do we discover?](https://www.youtube.com/watch?v=Un0RcTMPJ64)
    [Deriving a Rule for Differentiating Powers of x](https://www.youtube.com/watch?v=I8IM9P-2TRU)
    [1 of 3: Introducing a substitution](https://www.youtube.com/watch?v=U0m4MsOgETw)
    [2 of 3: Combining derivatives](https://www.youtube.com/watch?v=z-tEsz0bSrA)
    [How To Understand Derivatives: The Product, Power & Chain Rules](https://betterexplained.com/articles/derivatives-product-power-chain/)
    [Product Rule](https://en.wikipedia.org/wiki/Product_rule)
    [Common Derivatives and Integrals](https://www.coastal.edu/media/academics/universitycollege/mathcenter/handouts/calculus/deranint.PDF)
    [Introduction to partial derivatives](https://mathinsight.org/partial_derivative_introduction)
    [Partial derivatives - How to solve?](https://www.youtube.com/watch?v=rnoToCoEK48)
    [Integral](https://en.wikipedia.org/wiki/Integral)
    [Integration and the fundamental theorem of calculus](https://www.youtube.com/watch?v=rfG8ce4nNh0)
    [Introduction to Integration](https://www.mathsisfun.com/calculus/integration-introduction.html)
    [Indefinite Integral - Basic Integration Rules, Problems, Formulas, Trig Functions, Calculus](https://www.youtube.com/watch?v=o75AqTInKDU)
    [Definite Integrals](https://www.mathsisfun.com/calculus/integration-definite.html)
    [Definite Integral](https://www.youtube.com/watch?v=Gc3QvUB0PkI)
    [Multiple integral](https://en.wikipedia.org/wiki/Multiple_integral)
    [Double integral 1](https://www.youtube.com/watch?v=85zGYB-34jQ)
    [Double integrals 2](https://www.youtube.com/watch?v=TdLD2Zh-nUQ)

---
## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
### General
*    Summation and Product notation
*    What is a series?
*    Common series
*    What is a derivative?
*    What is the product rule?
*    What is the chain rule?
*    Common derivative rules
*    What is a partial derivative?
*    What is an indefinite integral?
*    What is a definite integral?
*    What is a double integral?

## Requirements
### Multiple Choice Questions

*    Allowed editors: vi, vim, emacs
*    Type the number of the correct answer in your answer file
*    All your files should end with a new line

Example:

What is 9 squared?
*    99
*    81
*    3
*    18
```
alexa@ubuntu$ cat answer_file
2
alexa@ubuntu$
```
---
## Python Scripts
*    Allowed editors: vi, vim, emacs
*    All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
*    All your files should end with a new line
*    The first line of all your files should be exactly #!/usr/bin/env python3
*    A README.md file, at the root of the folder of the project, is mandatory
*    Your code should use the pycodestyle style (version 2.5)
*    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
*    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
*    Unless otherwise noted, you are not allowed to import any module
*    All your files must be executable
*    The length of your files will be tested using wc

---
### [0. Sigma is for Sum](./0-sigma_is_for_sum)

[\sum_{i=2}^{5} i]
```
    3 + 4 + 5
    3 + 4
    2 + 3 + 4 + 5
    2 + 3 + 4
```
---

### [1. The Greeks pronounce it sEEgma](./1-seegma)

[\sum_{k=1}^{4} 9i - 2k]
```
    90 - 20
    36i - 20
    90 - 8k
    36i - 8k
```
---

### [2. Pi is for Product](./2-pi_is_for_product)

[\prod_{i = 1}^{m} i]
```
    (m - 1)!
    0
    (m + 1)!
    m!
```
---

### [3. The Greeks pronounce it pEE](./3-pee)

[\prod_{i = 0}^{10} i]
```
    10!
    9!
    100
    0
```
---

### [4. Hello, derivatives!](./4-hello_derivatives)

[\frac{dy}{dx}] where [y = x^4 + 3x^3 - 5x + 1]
```
    [3x^3 + 6x^2 -4]
    [4x^3 + 6x^2 - 5]
    [4x^3 + 9x^2 - 5]
    [4x^3 + 9x^2 - 4]
```
---

### [5. A log on the fire](./5-log_on_fire)

[\frac{d (xln(x))}{dx}]
```
    [ln(x)]
    [\frac{1}{x} + 1]
    [ln(x) + 1]
    [\frac{1}{x}]
```
---

### [6. It is difficult to free fools from the chains they revere](./6-voltaire)

[\frac{d (ln(x^2))}{dx}]
```
    [\frac{2}{x}]
    [\frac{1}{x^2}]
    [\frac{2}{x^2}]
    [\frac{1}{x}]
```
---

### [7. Partial truths are often more insidious than total falsehoods](./7-partial_truths)

[\frac{\partial f(x, y)}{\partial y}] where [f(x, y) = e^{xy}] and [\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0]
```
    [e^{xy}]
    [ye^{xy}]
    [xe^{xy}]
    [e^{x}]
```
---

### [8. Put it all together and what do you get?](./8-all-together)

[\frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y})] where [\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0]
```
    [2x(1+y)e^{x^2y}]
    [2xe^{2x}]
    [2x(1+x^2y)e^{x^2y}]
    [e^{2x}]
```
---

### [9. Our life is the sum total of all the decisions we make every day, and those decisions areetermined by our priorities](./9-sum_total.py)
Write a function def summation_i_squared(n): that calculates [\sum_{i=1}^{n} i^2] :
*    n is the stopping condition
*    Return the integer value of the sum
*    If n is not a valid number, return None
*    You are not allowed to use any loops
```
alexa@ubuntu:0x02-calculus$ cat 9-main.py 
#!/usr/bin/env python3

summation_i_squared = __import__('9-sum_total').summation_i_squared

n = 5
print(summation_i_squared(n))
alexa@ubuntu:0x02-calculus$ ./9-main.py 
55
alexa@ubuntu:0x02-calculus$
```
---

### [10. Derive happiness in oneself from a good day's work](./10-matisse.py)
Write a function def poly_derivative(poly): that calculates the derivative of a polynomial:
*    poly is a list of coefficients representing a polynomial
*        the index of the list represents the power of x that the coefficient belongs to
*        Example: if [f(x) = x^3 + 3x +5] , poly is equal to [5, 3, 0, 1]
*    If poly is not valid, return None
*    If the derivative is 0, return [0]
*    Return a new list of coefficients representing the derivative of the polynomial
```
alexa@ubuntu:0x02-calculus$ cat 10-main.py 
#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))
alexa@ubuntu:0x02-calculus$ ./10-main.py 
[3, 0, 3]
alexa@ubuntu:0x02-calculus$
```
---

### [11. Good grooming is integral and impeccable style is a must](./11-integral)
```
    3x2 + C
    x4/4 + C
    x4 + C
    x4/3 + C
```
---

### [12. We are all an integral part of the web of life](./12-integral)
```
    e2y + C
    ey + C
    e2y/2 + C
    ey/2 + C
```
---

### [13. Create a definite plan for carrying out your desire and begin at once](./13-definite)
```
    3
    6
    9
    27
```
---

### [14. My talents fall within definite limitations](./14-definite)
```
    -1
    0
    1
    undefined
```
---

### [15. Winners are people with definite purpose in life](./15-definite)
```
    5
    5x
    25
    25x
```
---

### [16. Double whammy](./16-double)
```
    9ln(2)
    9
    27ln(2)
    27
```
---

### [17. Integrate](./17-integrate.py)

Write a function def poly_integral(poly, C=0): that calculates the integral of a polynomial:
*    poly is a list of coefficients representing a polynomial
*        the index of the list represents the power of x that the coefficient belongs to
*        Example: if [f(x) = x^3 + 3x +5] , poly is equal to [5, 3, 0, 1]
*    C is an integer representing the integration constant
*    If a coefficient is a whole number, it should be represented as an integer
*    If poly or C are not valid, return None
*    Return a new list of coefficients representing the integral of the polynomial
*    The returned list should be as small as possible
```
alexa@ubuntu:0x02-calculus$ cat 17-main.py 
#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
alexa@ubuntu:0x02-calculus$ ./17-main.py 
[0, 5, 1.5, 0, 0.25]
alexa@ubuntu:0x02-calculus$
```
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
