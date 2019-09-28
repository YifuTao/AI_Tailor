from visdom import Visdom
import numpy as np
viz = Visdom()

win = viz.line(Y=np.array([0]))
viz.line(
        X=np.array([0]),
        Y=np.array([0.5]),
        name='train',
        win=win,
        update='append'
    )
viz.line(
        X=np.array([0]),
        Y=np.array([0.3]),
        name='val',
        win=win,
        update='append'
    )
viz.line(
        X=np.array([1]),
        Y=np.array([0.3]),
        name='train',
        win=win,
        update='append'
    )
viz.line(
        X=np.array([1]),
        Y=np.array([0.2]),
        name='val',
        win=win,
        update='append'
    )
'''
viz.line(
        X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
        Y=np.column_stack((np.linspace(5, 10, 10),
                           np.linspace(5, 10, 10) + 5)),
        win=win,
        update='append'
    )

win = viz.line(
        X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
        Y=np.column_stack((np.linspace(5, 10, 10),
                           np.linspace(5, 10, 10) + 5)),
    )

viz.line(
        X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
        Y=np.column_stack((np.linspace(5, 10, 10),
                           np.linspace(5, 10, 10) + 5)),
        win=win,
        update='append'
    )
viz.line(
        X=np.arange(21, 30),
        Y=np.arange(1, 10),
        win=win,
        name='1.5',
        update='append'
    )
viz.line(
        X=np.arange(1, 10),
        Y=np.arange(11, 20),
        win=win,
        name='delete this',
        update='append'
    )
a=input()
viz.line(
        X=np.arange(1, 10),
        Y=np.arange(11, 20),
        win=win,
        name='4',
        update='insert'
    )
'''