# standard library
import unittest
import sys

import numpy as np
from numpy.testing import assert_allclose

# utilities
from cpymad.madx import Madx, CommandLog

# Module under test:
import madgui.online.ovm


def _compute_initial_position_alt_1(A, a, B, b):
    """
    This is an alternative implementation to _compute_initial_position
    using my original (overcomplicated) line of thinking...
    """
    Z = np.zeros
    I = np.eye
    rows = (0,2)
    cols = (0,1,2,3,6)
    AL = A[rows,:][:,cols]
    BR = B[rows,:][:,cols]
    M = np.bmat([
        [AL,        Z((2,5))],
        [Z((2,5)),  BR      ],
        [I(1,5,4),  Z((1,5))],
        [I(5),      -I(5)   ],
    ])
    m = np.hstack((a, b, 1, Z(5)))
    return np.linalg.lstsq(M, m)[0][:4]


def _compute_initial_position_alt_2(A, a, B, b):
    """
    This is an alternative implementation to _compute_initial_position
    using Rainer's line of thought.
    """
    rows = cols = (0,1,2,3,6)
    AL = A[rows,:][:,cols]
    BL = B[rows,:][:,cols]
    AR = np.array([
        [ 0,  0, 0, 0],
        [-1,  0, 0, 0],
        [ 0,  0, 0, 0],
        [ 0, -1, 0, 0],
        [ 0,  0, 0, 0],
    ])
    BR = np.array([
        [0, 0,  0,  0],
        [0, 0, -1,  0],
        [0, 0,  0,  0],
        [0, 0,  0, -1],
        [0, 0,  0,  0],
    ])
    M = np.bmat([[AL, AR],
                 [BL, BR]])
    m = np.array([a[0], 0, a[1], 0, 1, b[0], 0, b[1], 0, 1])
    return np.linalg.lstsq(M, m)[0][:4]


class TestOVM(unittest.TestCase):

    par = ['x', 'px', 'y', 'py', 't', 'pt']
    val = [+0.0010, -0.0015, -0.0020, +0.0025, +0.0000, +0.0000]
    twiss = {'betx': 0.0012, 'alfx': 0.0018,
             'bety': 0.0023, 'alfy': 0.0027}
    twiss.update(zip(par, val))

    def _mad(self, doc):
        mad = Madx(command_log=CommandLog(sys.stdout, 'X:> '))
        for line in doc.splitlines():
            mad._libmadx.input(line)
        return mad

    def setUp(self):
        self.mad = self._mad("""
            seq: sequence, l=10, refer=entry;
                q1: QUADRUPOLE, K1:=K1_Q1, at=3, l=1;
                k1: HKICKER, KICK=0.01, at=5;
                q2: QUADRUPOLE, K1:=K1_Q2, at=6, l=1;
            endsequence;
            beam;
        """)

    def test_compute_initial_position(self):

        kw = {'sequence': 'seq',
              'twiss_init': self.twiss}

        self.mad.globals['K1_Q1'] = 0.001
        tw1 = self.mad.twiss(**kw)
        A = self.mad.get_transfer_map_7d(**kw)
        a = (tw1['x'][-1], tw1['y'][-1])

        self.mad.globals['K1_Q1'] = 0.003
        tw2 = self.mad.twiss(**kw)
        B = self.mad.get_transfer_map_7d(**kw)
        b = (tw2['x'][-1], tw2['y'][-1])

        x_actual = madgui.online.ovm._compute_initial_position(A, a, B, b)
        x_alt_1 = _compute_initial_position_alt_1(A, a, B, b)
        x_alt_2 = _compute_initial_position_alt_2(A, a, B, b)

        assert_allclose(x_actual, x_alt_1)
        assert_allclose(x_actual, x_alt_2)
        assert_allclose(x_actual, self.val[:4])


if __name__ == '__main__':
    unittest.main()
