# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from typing import Union
import numpy as np

from ..utils import str_to_arr


class Lookup():
    """
    Compares depth to prerendered poses.

    Settings defined globally instead of instantially.

    Num rendered and joints changed can be modified.
    """
    pass

class BaseStage():
    def __init__(self, to_render: int):
        self.to_render = to_render

class SFlip(BaseStage):
    def __init__(self, to_render: int):
        """
        Perspecitive flip the S joint.
        'Flip' amount depends on camera position. Will produce roughly the same shadow. 

        Parameters
        ----------
        to_render : int
            Number of links to render
        """
        super().__init__(to_render)

class Sweep(BaseStage):
    def __init__(self, to_render: int, divs: int, joints: Union[str, np.ndarray], range: float = None):
        """Base Sweep Class"""
        super().__init__(to_render)
        self.divs, self.range = divs, range
        self.joints = str_to_arr(joints) if type(joints) is str else joints

class InterpolativeSweep(Sweep):
    def __init__(self, to_render: int, divs: int, joints: Union[str, np.ndarray], range: float = None):
        """
        Sweep through a joint's range in a specified number of divisons.
        Attempts to interpolate error function to find true minimum of loss.

        See TensorSweep for similar behavior.

        Parameters
        ----------
        to_render : int
            Number of links to render
        divs : int
            Divisons of the range to do
        joints : Union[str, np.ndarray]
            Joints to sweep
        range : float, optional
            Range in rad to sweep about current pos, or None to do entire joint range, by default None.
        """
        super().__init__(to_render, divs, joints, range=range)

class TensorSweep(Sweep):
    def __init__(self, to_render: int, divs: int, joints: Union[str, np.ndarray], range: float = None):
        """
        Sweep through a joint's range in a specified number of divisons.
        Uses GPU matrix acceleration to determine loss. InterpolativeSweep may give better results in most cases.

        See InterpolativeSweep for similar behavior.

        Parameters
        ----------
        to_render : int
            Number of links to render
        divs : int
            Divisons of the range to do
        joints : Union[str, np.ndarray]
            Joints to sweep
        range : float, optional
            Range in rad to sweep about current pos, or None to do entire joint range, by default None.
        """
        super().__init__(to_render, divs, joints, range=range)

class Descent(BaseStage):
    def __init__(self, to_render: int, iterations: int, joints: Union[str, np.ndarray],
        init_rate: Union[float, int, np.ndarray] = None, 
        rate_reduction: float = 0.5, early_stop_thresh: float = 0.01
        ):
        """[summary]

        Parameters
        ----------
        to_render : int
            Number of links to render
        iterations : int
            Max number of iterations to complete
        joints : Union[str, np.ndarray]
            Joints to use for descent
        init_rate : Union[float, int, np.ndarray], optional
            Default step size for each joint (six entries) None on one or all will continue past descent rate.
            A single value, if given, will apply to all entries, by default None
        rate_reduction : float, optional
            Scalar to apply to step size/rate whenever no progress is made with steps, by default 0.5
        early_stop_thresh : float, optional
            Percentage change of error. If below this value, descent will end prematurely, by default 0.01
        """

        super().__init__(to_render)
        self.its, self.rate_redux, self.early_stop = iterations, rate_reduction, early_stop_thresh
        self.joints = str_to_arr(joints) if type(joints) is str else joints
        self.init_rate = [init_rate]*6 if type(init_rate) in [float, int] or init_rate is None else init_rate

# Class Aliases
IntSweep = InterpolativeSweep
ISweep = InterpolativeSweep
TSweep = TensorSweep



def getStages(angles:str):
    """Return stages for a given angle prediction set

    Parameters
    ----------
    angles : str
        Angles to predict ('SL','SLU', etc)
    """


    if angles == 'SL':

        lookup = Lookup()
        s_flip = SFlip(4)
        s_sweep_narrow = InterpolativeSweep(4,10,'S',0.1)
        l_sweep_narrow = InterpolativeSweep(4,10,'L',0.1)

        sl_fine_tune = Descent(4,40,'SL',[.05,.05,None,None,None,None], early_stop_thresh = .0075)

        sweeps = [l_sweep_narrow,s_sweep_narrow]

        # return [lookup, s_flip, *sweeps, s_flip, *sweeps, s_flip, sl_fine_tune]
        return [lookup, s_flip, *sweeps, s_flip]

    elif angles == 'SLU':

        lookup = Lookup()
        s_flip_4 = SFlip(4)
        sl_tune = Descent(4,10,'SL',[0.05,0.05,0.1,0.5,0.5,0.5],early_stop_thresh=0.1)

        sl_init = [s_flip_4, sl_tune, s_flip_4]

        u_sweep_wide = InterpolativeSweep(6,25,'U')
        s_flip_6 = SFlip(6)
        u_sweep_narrow = InterpolativeSweep(6, 10, 'U',0.1)
        
        u_stages = [u_sweep_wide, s_flip_4, s_flip_6, u_sweep_narrow]
        
        full_tune = Descent(6,40,'SLU',early_stop_thresh=0.0075)

        return [lookup, *sl_init, *u_stages, full_tune]

    elif angles == 'SLUB':
        # Not currently defined
        pass

    elif angles == 'SLURB':
        # Not currently defined
        pass

    return None
