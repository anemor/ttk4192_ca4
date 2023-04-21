(define (problem move_robot) (:domain ca4_test)
(:objects
    turtlebot0 - robot
    battery0 - battery
    camera0 - camera

    charger0 charger1 charger2 - charger
    pump0 pump1 - pump
    valve0 valve1 - valve
    WP0 WP1 WP2 WP3 WP4 WP5 WP6 - waypoint
)

(:init
    (no-check valve0) (no-check valve1) (no-check pump0) (no-check pump1)
    (at turtlebot0 WP4) (no-bat battery0)

    (at pump0 WP5) (at pump1 WP6)
    (at valve0 WP1) (at valve1 WP2)
    (at charger0 WP0) (at charger1 WP4) (at charger2 WP3)

    (onboard camera0 turtlebot0) (onboard battery0 turtlebot0)

    (path WP0 WP1) (path WP0 WP2) (path WP0 WP3) (path WP0 WP4) (path WP0 WP5) (path WP0 WP6)
    (path WP1 WP2) (path WP1 WP3) (path WP1 WP4) (path WP1 WP6) (path WP1 WP0) ;(path WP1 WP5)
    (path WP2 WP3) (path WP2 WP4) (path WP2 WP5) (path WP2 WP6) (path WP2 WP0) (path WP2 WP1)
    (path WP3 WP4) (path WP3 WP5) (path WP3 WP6) (path WP3 WP0) (path WP3 WP1) (path WP3 WP2)
    (path WP4 WP6) (path WP4 WP0) (path WP4 WP1) (path WP4 WP2) (path WP4 WP3) ;(path WP4 WP5)
    (path WP5 WP6) (path WP5 WP0) (path WP5 WP2) (path WP5 WP3) (path WP5 WP4) ;(path WP5 WP1) 
    (path WP6 WP0) (path WP6 WP1) (path WP6 WP2) (path WP6 WP3) (path WP6 WP4) (path WP6 WP5)
)

(:goal (and
    (at turtlebot0 WP3) (check valve0) (check valve1) (check pump0) (check pump1) (full-bat battery0)
))
)