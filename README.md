===
erfa_numpy
===

vectorized erfa library with numpy


===
requirements
===

numpy version > 1.6 

===
installation
===
  ::
      
      python setup.py install

===
examples
===

  ::
      
      >>> import erfa
      >>> import numpy as np
      >>> pnat = np.array([(-0.76321968546737951,-0.60869453983060384,-0.21676408580639883)])
      >>> v = np.array([(2.1044018893653786e-5,-8.9108923304429319e-5,-3.8633714797716569e-5)])
      >>> s = np.array([0.99980921395708788])
      >>> bm1 = np.array([0.99999999506209258])
      >>> ppr = erfa.ab(pnat, v, s, bm1)
      >>> ppr
      array([[-0.76316311, -0.60875531, -0.21679263]])
      >>> d = np.array([1957.3, 2014.5])
      >>> erfa.besselian_epoch_jd(d)
      (array([ 2400000.5,  2400000.5]), array([ 35948.19151015,  56840.04528042]))
      
