# 0x05. Advanced Linear Algebra

---
## Read or watch:
*   [The determinant | Essence of linear algebra](/rltoken/F61LWMDnv216TI4VURIavQ "The determinant | Essence of linear algebra")
*   [Determinant of a Matrix](/rltoken/d_uY4uoRqk7EU4Pg9qz7Ig "Determinant of a Matrix")
*   [Determinant](/rltoken/eLZ_J1BAltkAFuNsOhutKQ "Determinant")
*   [Determinant of an empty matrix](/rltoken/5REQT4orNW4FNpoDdy0RLg "Determinant of an empty matrix")
*   [Inverse matrices, column space and null space](/rltoken/5alpJ5Uyeb8QWaWmKeIFcA "Inverse matrices, column space and null space")
*   [Inverse of a Matrix using Minors, Cofactors and Adjugate](/rltoken/nAtBpV2gGB1PhMtfw8JWww "Inverse of a Matrix using Minors, Cofactors and Adjugate")
*   [Minor](/rltoken/IAjFHGF9pgUlajyVY0ytaA "Minor")
*   [Cofactor](/rltoken/KykXUKmSVLjYmT7z_Nc07g "Cofactor")
*   [Adjugate matrix](/rltoken/30NiZqFhPtoe61DUmXkUCA "Adjugate matrix")
*   [Singular Matrix](/rltoken/Wdpp4xtLOgxFq5kGyBpZsw "Singular Matrix")
*   [Elementary Matrix Operations](/rltoken/MRQqIv0dCyRivrpld6_5TQ "Elementary Matrix Operations")
*   [Gaussian Elimination](/rltoken/VTo7Q09EqloHCv-q6XGw3g "Gaussian Elimination")
*   [Gauss-Jordan Elimination](/rltoken/CMLsBAfm0W4uqe2r_Oezkw "Gauss-Jordan Elimination")
*   [Matrix Inverse](/rltoken/2Oz0LITPFPX9PTNn9zqrwQ "Matrix Inverse")
*   [Eigenvectors and eigenvalues | Essence of linear algebra](/rltoken/E8_Gy9NO7Gjb4LxN0jUFWQ "Eigenvectors and eigenvalues | Essence of linear algebra")
*   [Eigenvalues and eigenvectors](/rltoken/n1w31RZHRl8_tNhUY1-TRA "Eigenvalues and eigenvectors")
*   [Eigenvalues and Eigenvectors](/rltoken/ZSkHYcwTonsySFhCBrM8uA "Eigenvalues and Eigenvectors")
*   [Definiteness of a matrix](/rltoken/qODLYtIIrUMQLsEENialNQ "Definiteness of a matrix") **Up to Eigenvalues**
*   [Definite, Semi-Definite and Indefinite Matrices](/rltoken/cuiQxdTiqu_4T4k2kVVelQ "Definite, Semi-Definite and Indefinite Matrices") **Ignore Hessian Matrices**
*   [Tests for Positive Definiteness of a Matrix](/rltoken/rYtty7OT1dlzKac_-QGPhA "Tests for Positive Definiteness of a Matrix")
*   [Positive Definite Matrices and Minima](/rltoken/S-Yndhr3Pr-rTmzGgiIHTQ "Positive Definite Matrices and Minima")
*   [Positive Definite Matrices](/rltoken/j-ejEoIpd7Cg6SxMlChpew "Positive Definite Matrices")
*   [numpy.linalg.eig](/rltoken/AiJWrZH4RCkUydqzEeut-A "numpy.linalg.eig")

---
### Description 
* What is a determinant? How would you calculate it?
* What is a minor, cofactor, adjugate? How would calculate them?
* What is an inverse? How would you calculate it?
* What are eigenvalues and eigenvectors? How would you calculate them?
* What is definiteness of a matrix? How would you determine a matrix’s definiteness?

---
## General Requirements
*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.5)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise noted, you are not allowed to import any module
*   All your files must be executable
*   The length of your files will be tested using `wc`

---
## Files
| File | Description |
| ------ | ------ |
| [0-determinant.py](0-determinant.py) | Function determinant that calculates the determinant of a matrix. |
| [1-minor.py](1-minor.py) | Function minor that calculates the minor matrix of a matrix. |
| [2-cofactor.py](2-cofactor.py) | Function cofactor that calculates the cofactor matrix of a matrix. |
| [3-adjugate.py](3-adjugate.py) | Function adjugate that calculates the adjugate matrix of a matrix. |
| [4-inverse.py](4-inverse.py) | Function inverse that calculates the inverse of a matrix. |
| [5-definiteness.py](5-definiteness.py) | Function definiteness that calculates the definiteness of a matrix. |

---
## Build with
- Python (python 3.5)
- Numpy (numpy 1.15)
- Ubuntu 16.04 LTS 

---
### [0. Determinant](./0-determinant.py)
Write a function `def determinant(matrix):` that calculates the determinant of a matrix:
*   `matrix` is a list of lists whose determinant should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square, raise a `ValueError` with the message `matrix must be a square matrix`
*   The list `[[]]` represents a `0x0` matrix
*   Returns: the determinant of `matrix`
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./0-main.py 
    1
    5
    -2
    0
    192
    matrix must be a list of lists
    matrix must be a square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```

### [1. Minor](./1-minor.py)
Write a function `def minor(matrix):` that calculates the minor matrix of a matrix:
*   `matrix` is a list of lists whose minor matrix should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the minor matrix of `matrix`
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./1-main.py 
    [[1]]
    [[4, 3], [2, 1]]
    [[1, 1], [1, 1]]
    [[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```

### [2. Cofactor](./2-cofactor.py)
Write a function `def cofactor(matrix):` that calculates the cofactor matrix of a matrix:
*   `matrix` is a list of lists whose cofactor matrix should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the cofactor matrix of `matrix`
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./2-main.py 
    [[1]]
    [[4, -3], [-2, 1]]
    [[1, -1], [-1, 1]]
    [[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```

### [3. Adjugate](./3-adjugate.py)
Write a function `def adjugate(matrix):` that calculates the adjugate matrix of a matrix:
*   `matrix` is a list of lists whose adjugate matrix should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the adjugate matrix of `matrix`
```   
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./3-main.py 
    [[1]]
    [[4, -2], [-3, 1]]
    [[1, -1], [-1, 1]]
    [[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```

### [4. Inverse](./4-inverse.py)
Write a function `def inverse(matrix):` that calculates the inverse of a matrix:
*   `matrix` is a list of lists whose inverse should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the inverse of `matrix`, or `None` if `matrix` is singular
``` 
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./4-main.py 
    [[0.2]]
    [[-2.0, 1.0], [1.5, -0.5]]
    None
    [[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```

### [5. Definiteness](./5-definiteness.py)
Write a function `def definiteness(matrix):` that calculates the definiteness of a matrix:
*   `matrix` is a `numpy.ndarray` of shape `(n, n)` whose definiteness should be calculated
*   If `matrix` is not a `numpy.ndarray`, raise a `TypeError` with the message `matrix must be a numpy.ndarray`
*   If `matrix` is not a valid matrix, return `None`
*   Return: the string `Positive definite`, `Positive semi-definite`, `Negative semi-definite`, `Negative definite`, or `Indefinite` if the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectively
*   If `matrix` does not fit any of the above categories, return `None`
*   You may `import numpy as np`
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./5-main.py 
    Positive definite
    Positive semi-definite
    Negative semi-definite
    Negative definite
    Indefinite
    None
    None
    matrix must be a numpy.ndarray
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
---
## Author

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
