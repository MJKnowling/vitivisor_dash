# vitivisor_dash

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MJKnowling/vitivisor_dash.git/master?filepath=skinny_vitivisor_dash.ipynb)

This repository is under development as part of the VitiVisor project. It is designed to showcase some of the prediction and advisory aspects of the project in a streamlined fashion.

## Instructions:

Launch the Binder using a link provided or by clicking the `launch binder` badge above. This may take a little while. We are working to reduce the time this takes.

Then click on the `Appmode` button in the Jupyter menu. This step will soon happen automatically.

Once the dashboard has loaded, you will see two section headings. The first is `Management Scenarios` and the second is `Seasonal projections`.

Under `Management Scenarios`, simply select which scenarios you would like to explore. Three scenarios have been prepared earlier for you. The scenario names correspond to those in the scenario input files (`Scenario01.json`, `Scenario02.json` and `Scenario03.json`). All of the data fields in these files can be (and are designed to be) modified. To do so, go back to the menu by clicking `jupyter` at the top of the page (or right-clicking and opening link in new tab for convenience) and click on the scenario input json files that you want to modify. A new tab will open. The `Value` fields relating to various biophysical and financial variables can be modified therein. Once complete, be sure to save the file. Then return to the dashboard interface and follow below steps.

Once you have defined and selected the scenarios that you wish to investigate, click `Evaluate Scenario(s)`. 

Then simply select from the drop-down menu what aspect(s) of your vineyard you wish to consider, including `how my vine is developing` (e.g., canopy growth, end-of-season yield, key phenological dates), `where my water is going` (e.g., irrigation, root uptake) and `what my financial position is` (e.g., crush revenue, irrigation and canopy and disease management costs, bottom line).
