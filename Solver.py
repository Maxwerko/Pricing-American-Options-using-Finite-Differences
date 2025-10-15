"""
=====================================================
Finite Difference solver to price American options
=====================================================

This script Implements both PSOR and the operator splitting method presented 
in the article and compare them. Note that in the first line of equation (10)
the signs in front of λ should be the opposite when implementing the operator 
splitting method (it's a typo in the article). 
"""