# Explicit Runge-Kutta methods

A_euler = (
    (),
)

b_euler = (
    1.,
)

A_midpoint = (
    (),
    (.5,)
)

b_midpoint = (
    0.,
    1.
)

A_rk3 = (
    (),
    (.5,),
    (-1., 2.)
)

b_rk3 = (
    1./6,
    2./3,
    1./6
)

A_rk4 = (
    (),
    (.5,),
    (0., .5),
    (0., 0., 1.)
)

b_rk4 = (
    1./6,
    1./3,
    1./3,
    1./6
)