# HELlo Firebreak - D*Lite Version

Wildfire simulation for testing real-time tractor path planning (firebreak cutting) on farmland using D*Lite.

## What’s Implemented

- 2D grid farm with elevation + uniform fuel (soybean field)
- Rothermel-based fire spread (wind, moisture, slope, burn duration)
- Tractor agent that:

  - moves in 4 directions
  - creates firebreaks (non-burnable cells)
  - can exit safely or die if entering fire
- Environment API for RL & D*: `reset()`, `step()`, `render()`
- State codes:

  - 0 = fuel
  - 1 = burning
  - 2 = burned
  - 3 = firebreak
  - 4 = tractor
- `env.demo()` included for quick visual run

## Notes for Teammates

- The `demo()` is a simple example, and is a good place to start.
- `env.py` contains all helper functions for accessing:
  - grid state
  - fire spread updates
  - tractor movement + firebreak logic

In `env.py` the logic of fire:


| Rule                         | Description                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| **Fire Spread Model**        | Uses a Rothermel-based model (wind, moisture, slope, cell fuel).                 |
| **States of a Cell**         | **0** fuel, **1** burning, **2** burned, **3** firebreak.                        |
| **Burn Duration**            | A cell burns for\~3 minutes (configurable) before becoming **burned (state 2)**. |
| **Firebreaks Stop Fire**     | Fire cannot ignite or pass through **firebreak** cells.                           |
| **Burned Cells Stay Burned** | Once burned, a cell stays burned until the end of the simulation.                |

In `env.py` the logic of tractor:


| Rule                          | Description                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| **Movement**                  | Tractor moves 1 cell per step in 4 directions: up/down/left/right (no diagonal). Currently will try and stay between a stated "safe zone" - close enough to fire for realistic use but far enough to save land (configurable values)     |
| **Firebreak**                 | Every cell the tractor drives through becomes a**firebreak** (unburnable).           |
| **Cannot Occupy Fire**        | If the tractor enters a**burning** cell → **dies immediately**.                     |
| **Cannot Occupy Burned Land** | If the tractor enters a**burned** cell (already burned-out) → **dies immediately**. |
| **Exiting Map**               | ~~If tractor leaves the map boundary → it**survives** and the tractor is removed.~~ Currently it doesn't leave the map     |
| **No Respawn**                | Once dead or exited, tractor no longer moves.                                        |
| **Episode Ends on Death**     | The entire simulation stops instantly when tractor dies.                             |
| **Goal Endpoint**             | Tractor should aim for the bottom cell closest to the center of the fire             |


End condition:

| Condition                 | Meaning                                             |
| ------------------------- | --------------------------------------------------- |
| **Tractor Dies**          | Episode ends instantly.                             |
| **No More Burning Cells** | Fire reached a steady state (fully burned out).     |
| **Max Steps Reached**     | Safety cap to avoid infinite loops (mainly for RL). |

## Run the Simulation

Requirements:

```
pip install numpy matplotlib
```

If using conda:

```
conda activate wildfire
python3 sim.py
```

## Scale Assumptions

- 1 cell = 6 ft × 6 ft **~~shouldn't this be 12ft x 12ft~~**
- Tractor tiller ≈ 12 ft wide (currently simplified to 1-cell width)
- Burns ~3 minutes per cell (default, simplify version)
- Tractor moves 6ft-8.5ft (1 cell) per minute (0.06-0.09 mph) [shortest distance: 6ft, longest distance 8.5 ft on diagonal]

## File Structure

* env.py # main environment (D*Lite + demo)
* fire.py # fire spread model
* grid.py # grid + burning/burned state logic
* tractor.py # tractor movement + firebreak
* mapgen.py # farm map generator
* sim.py # example run legacy

## Next Steps

- D* planner integration into DQN model
- RL reward design + Gym wrapper
- Fuel model variations (beyond soy)

## Todo: 
- Fix waypointing so that they don't just randomly wander off when reach grid edge
- Fix tractor ending --> exit field upon reaching end of waypoints
- Figure out a way to determine of a waypoint is cornered in by the fire (unreachable) and remove it from waypoint direction list
- 
