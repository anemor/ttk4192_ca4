(define (domain ca4_test)
    (:requirements :strips :typing)
    (:types
        valve pump - checkobj
        valve pump charger robot - atobj    ; At object
        battery camera - onbobj             ; Onboard object
        robot battery camera charger pump valve waypoint
    )

    (:predicates
        (at ?obj - atobj ?wp - waypoint)
        (onboard ?obj - onbobj ?r - robot)
        (no-check ?obj - checkobj)
        (check ?obj - checkobj)
        (no-bat ?b - battery)
        (full-bat ?b - battery)
        (path ?x - waypoint ?y - waypoint)
    )

    (:action move
        :parameters (?from - waypoint ?to - waypoint ?r - robot)
        :precondition ( and (at ?r ?from) (path ?from ?to) )
        :effect ( and (at ?r ?to) (not (at ?r ?from)) )
    )

    (:action take-picture
        :parameters (?wp - waypoint ?p - pump ?r - robot ?c - camera)
        :precondition ( and (onboard ?c ?r) (at ?p ?wp) (at ?r ?wp) (no-check ?p) )
        :effect ( and (check ?p) (not (no-check ?p)) )
    )

    (:action inspect-valve
        :parameters (?wp - waypoint ?v - valve ?r - robot)
        :precondition ( and (at ?v ?wp) (at ?r ?wp) (no-check ?v) )
        :effect ( and (check ?v) (not (no-check ?v)) )
    )

    (:action charge-robot
        :parameters (?wp - waypoint ?c - charger ?r - robot ?b - battery)
        :precondition ( and (onboard ?b ?r) (at ?r ?wp) (at ?c ?wp) (no-bat ?b))
        :effect ( and (full-bat ?b) (not (no-bat ?b)) )
    )
)