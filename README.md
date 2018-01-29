# scioly-team-assigner
This python script chooses and assigns to events a team of 15 people from a list of candidates. It optimizes the sum of their test scores.

## Getting Started
Make a new folder to put everything in. Then download everything in the Github repo. To run, open a command line and navigate to the folder. Then run 'python team_assigner.py' followed by the names of the csv files with data in them:
'''
python team_assigner.py example_scores.csv
'''
[[Pictures/running_the_team_assigner.png|alt=how to run the team assigner]]
Replace example_scores.csv with whatever the names of your score sheets are. There can be multiple csv files included.

### Needed packages
* Python (The latest 3.x is the supported version, but it will probably run with Python 2.7).
* Numpy. A fast computation package for Python not included in the standard distribution (but very widely used).
* Scipy library. This supplies the Hungarian algorithm used to assign people to events.

####Checking your Python configuration:

Open a command line. Run:
'''
python
'''
When the Python shell loads, run:
'''
import numpy
from scipy.optimize import linear_sum_assignment
'''

If either of these commands causes an error, you don't have that package installed. A successful test does nothing, like this:
[[Pictures/check_python.png|alt=successful python test]]

### Output
The teams the assigner comes up with are put in a text file, team_config.txt. They are sorted by score. Duplicates are removed.

#### Loading prior generated teams
The assigner also produces an 'outfile' (no extension). This is a non-human-readable file that stores the output teams so that, when the assigner is next run, they can be loaded in and included in the list of teams generated.

Each time the team assigner is restarted, it loads in the teams in the outfile and sees if they can be optimized further. If the input data has not changed, this will not be possible. If it has, the assigner will iterate over all the teams and see if they can be further optimized. This will take 15-30 seconds per team (depending on the amount of optimization possible).