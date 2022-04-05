import streamlit as st

from mavis.db import WorkflowDAO


class WorkflowWidget:
    def __init__(self):
        from mavis.ui.widgets.main import ModuleWidget
        from mavis.ui.widgets.utils import ListView
        self.dao = WorkflowDAO()
        
        st.write("### Workflows ")
        
        if not st.checkbox("Activate Workflow: " + self.dao.get()):
            workflows = self.dao.get_all()

            selection = st.selectbox(
                "Select Workflow",
                workflows,
                workflows.index(self.dao.get()),
            )

            if st.button("Select", key="SelectWorkflow "):
                self.dao.activate(selection)

            if st.button("Delete Workflow " + self.dao.get()):
                self.dao.delete()

            if st.checkbox("Edit Workflow"):
                st.title("Edit Workflow")
                st.write("Current Workflow: " + self.dao.get())
                pipelines = ListView(
                    list(ModuleWidget().get_modules()),
                    self.dao.workflow,
                    "Select Modules",
                ).selection()
                if st.button("Update Workflow"):
                    self.dao.set(self.dao.get(), pipelines)
            if st.checkbox("New Workflow"):
                st.title("New Workflow")
                name = st.text_input("Workflow Name", "Default")
                pipelines = st.multiselect(
                    "Select Modules",
                    list(ModuleWidget().get_modules())
                )
                if st.button("Create Workflow"):
                    self.dao.set(name, pipelines)

        else:
            from mavis import db
            from runpy import run_module

            pipeline = self.dao.iterate()
            db.BaseDAO.ACTIVE_PIPELINE = pipeline
            st.markdown(f"# {pipeline.split('.')[-1]}")
            run_module(pipeline)


