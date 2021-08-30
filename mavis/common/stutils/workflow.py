import shelve

import streamlit as st

from mavis.pkgutils import get_modules, execute
from mavis.shelveutils import workflow_path


class Workflow:
    def __init__(self, container):
        self.current = "Default"
        self.workflow = []
        self.pipeline = 0
        self.update()
        self.container = container

    def update(self):
        with shelve.open(workflow_path()) as d:
            # No current workflow
            if "current" not in d:
                d["current"] = "Default"

            self.current = d["current"]

            # No workflow db
            if "workflows" not in d:
                d["workflows"] = {"Default": []}
                self.workflow = []

            # Default workflow has been Deleted
            if "Default" not in d["workflows"]:
                workflows = d["workflows"]
                workflows.update({"Default": []})
                d["workflows"] = workflows

            # Workflow DB ready to use
            if self.current not in d["workflows"]:
                d["current"] = "Default"
                self.current = "Default"

            # Set current workflow
            self.workflow = d["workflows"][self.current]

            # No active pipeline
            if "pipeline" not in d:
                d["pipeline"] = 0
                self.pipeline = 0
            else:
                self.pipeline = d["pipeline"]

    def set(self, name, pipelines):
        with shelve.open(workflow_path()) as d:
            workflows = d["workflows"]
            workflows[name] = pipelines
            d["workflows"] = workflows
        self.activate(name)

    def activate(self, name):
        with shelve.open(workflow_path()) as d:
            d["current"] = name
            d["pipeline"] = 0
        self.update()
        st.info("Activated Workflow: " + self.current)

    def create(self):
        if self.container.checkbox("New Workflow"):
            st.title("New Workflow")
            name = st.text_input("Workflow Name", "Default")
            pipelines = st.multiselect("Select Pipelines",
                                       list(get_modules()))
            if st.button("Create Workflow"):
                self.set(name, pipelines)
                st.info("Created workflow" + self.current)

    def select(self):
        with shelve.open(workflow_path()) as d:
            workflows = list(d["workflows"].keys())
            selection = self.container.selectbox("Select Workflow",
                                             workflows,
                                             workflows.index(d["current"]))
        if self.container.button("Select"):
            self.activate(selection)

    def iterate(self):
        with shelve.open(workflow_path()) as d:
            if self.container.button("Next Pipeline"):
                self.pipeline = (self.pipeline + 1) % len(self.workflow)
                d["pipeline"] = self.pipeline
        if self.workflow:
            execute(self.workflow[(self.pipeline - 1)])
        else:
            st.warning("Workflow has no configures pipelines")

    def edit(self):
        if self.container.checkbox("Edit Workflow"):
            st.title("Edit Workflow")
            st.write("Current Workflow: " + self.current)
            pipelines = st.multiselect("Select Pipelines",
                                       list(get_modules()),
                                       self.workflow)
            if st.button("Update Workflow"):
                self.set(self.current, pipelines)

    def delete(self):
        if self.container.button("Delete Workflow " + self.current):
            with shelve.open(workflow_path()) as d:
                if self.current in d["workflows"]:
                    wf = d["workflows"]
                    del wf[self.current]
                    d["workflows"] = wf
                    self.container.info("Deleted Workflow " + self.current)
            self.activate("Default")

    def run(self):
        if not self.container.checkbox("Activate Workflow: " + self.current):
            self.select()
            self.delete()
            self.edit()
            self.create()
        else:
            self.iterate()