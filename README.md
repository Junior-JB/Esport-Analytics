# personal_project

Validation-first gameplay extraction scaffold.

Scope:

- scoreboard registration only
- killfeed extraction only
- utility counts only
- hardpoint score and hill only

Explicitly out of scope:

- advanced analytics
- recommendations
- predictive models
- ratings
- trend reporting

Module execution order:

1. Scoreboard Registration
2. Killfeed
3. Utility
4. Hardpoint

Key rule:

- scoreboard must complete registration before killfeed becomes active
- once registration succeeds, scoreboard disables itself

Project entry point:

- [app/main.py](/Users/juniorbenitez/Documents/main project/personal_project/app/main.py)

Config:

- [config/app_config.yaml](/Users/juniorbenitez/Documents/main project/personal_project/config/app_config.yaml)
