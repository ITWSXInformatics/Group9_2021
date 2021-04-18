# Group9_2021
#DeepChat

TOP REQUIREMENT: Need Google Colab for deployment test! (HEROKU Deployment failed, looking for alternatives)

Structure:
-------------
1. "static" folder contains CSS and JS files
2. "templates" folder contains index.html file'
3. app.py is the main app file with routes and logic
4. requirements.txt contains the dependency list from virtual env

Steps:
---------
1. Run the git configs with your email address and username
2. Clone the repo in your colab space (colab runtime doesn't save datafiles, so make sure to git push after you make any changes and leave the runtime idle)
3. cd to Group9_2021 folder
4. Install the dependencies from requirements.txt
5. Run the app - when you run it, first the training will be done, then there will be 3 URLs at the end. Click on the link with http://{something}.ngrok.io. It will open in a new tab and run the app. If you want to close, interrupt the execution with ctrl+c or from colab menu > Runtime > Interrupt Execution.
