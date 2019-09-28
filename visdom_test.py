import visdom
import numpy as np

viz = visdom.Visdom()
'''
viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

Y = np.linspace(-5, 5, 100)
viz.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)
'''
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
    name='2',
    update='append'
)

viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='delete this',
    update='append'
)

viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='4',
    update='insert'
)

viz.line(X=None, Y=None, win=win, name='delete this', update='remove')



win = viz.line(
    X=np.column_stack((
        np.arange(0, 10),
        np.arange(0, 10),
        np.arange(0, 10),
    )),
    Y=np.column_stack((
        np.linspace(5, 10, 10),
        np.linspace(5, 10, 10) + 5,
        np.linspace(5, 10, 10) + 10,
    )),
    opts={
        'dash': np.array(['solid', 'dash', 'dashdot']),
        'linecolor': np.array([
            [0, 191, 255],
            [0, 191, 255],
            [255, 0, 0],
        ]),
        'title': 'Different line dash types'
    }
)

viz.line(
    X=np.arange(0, 10),
    Y=np.linspace(5, 10, 10) + 15,
    win=win,
    name='4',
    update='insert',
    opts={
        'linecolor': np.array([
            [255, 0, 0],
        ]),
        'dash': np.array(['dot']),
    }
)

Y = np.linspace(0, 4, 200)
win = viz.line(
    Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
    X=np.column_stack((Y, Y)),
    opts=dict(
        fillarea=True,
        showlegend=False,
        width=800,
        height=800,
        xlabel='Time',
        ylabel='Volume',
        ytype='log',
        title='Stacked area plot',
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    ),
)
