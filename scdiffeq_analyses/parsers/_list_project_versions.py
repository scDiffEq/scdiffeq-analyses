def list_project_versions(project):
    versions = []
    for item in project.__dir__():
        if item.startswith("version"):
            try:
                versions.append(getattr(project, item))
            except Exception as e:
                print(f"Problem with: {item}: {e}")
    return versions
