upload and label your dataset, and get an API KEY here: https://app.roboflow.com/?model=undefined&ref=undefined
loading Roboflow workspace...
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipython-input-109607616.py in <cell line: 0>()
     34
     35 for name, version in datasets_info:
---> 36     project = rf.workspace().project(name)
     37     dataset = project.version(version).download("yolov8")
     38     print(f"✅ Dataset {name}-{version} downloaded to {dataset.location}")

/usr/local/lib/python3.12/dist-packages/roboflow/__init__.py in workspace(self, the_workspace)
    237
    238         if the_workspace is None:
--> 239             the_workspace = self.current_workspace
    240
    241         if self.api_key:  # Check if api_key was passed during __init__

AttributeError: 'Roboflow' object has no attribute 'current_workspace'
