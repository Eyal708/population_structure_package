# Population Structure Package
A package for performing transformation between fst, coalescence, and migration matrices.
The transformation between coalescence and migration is based on Wilkinson-Herbot's equations (2003).
The transformation between fst and coalescence is based on the Slatkin's equations (1991).
## Install package using pip
pip install population-structure
## Example usuage
```python
import population_structure.utils as psu
import numpy as np
m = np.array([[0, 1, 1], 
              [1, 0, 1], 
              [1, 1, 0]]) # A conservative migration matrix
t = psu.m_to_t(m) # The corresponding coalescence matrix according to W.H. (2003)
f = psu.m_to_f(m) # The corresponding fst matrix according to Slatkin (1991)
print(f"{t}\n{f}")
"""
prints:
[[3. 4. 4.]
 [4. 3. 4.]
 [4. 4. 3.]]
[[0.         0.14285714 0.14285714]
 [0.14285714 0.         0.14285714]
 [0.14285714 0.14285714 0.        ]]
"""
f = np.array([[0,0.1,0.2],
              [0.1,0,0.3],
              [0.2,0.3,0]]) # An fst matrix
psu.f_to_t(f) # Generates a possilbe corresponding coalescence matrix
psu.f_to_m(f) # Generate a possible corresponding migration matrix
```




