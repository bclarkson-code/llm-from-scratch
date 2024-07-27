import os


def create_rst_files(src_dir, docs_dir, package_name):
    # Ensure the docs directory exists
    os.makedirs(os.path.join(docs_dir, package_name), exist_ok=True)

    # Walk through the source directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Get the module name
                module_name = os.path.splitext(file)[0]

                # Create the full module path
                module_path = os.path.relpath(
                    os.path.join(root, file), src_dir
                )
                module_path = os.path.splitext(module_path)[0].replace(
                    os.path.sep, "."
                )

                # Create the RST content
                rst_content = f"""{module_name}
{'=' * len(module_name)}

.. automodule:: {package_name}.{module_path}
   :members:
   :undoc-members:
   :show-inheritance:
"""

                # Write the RST file
                rst_path = os.path.join(
                    docs_dir, package_name, f"{module_name}.rst"
                )
                with open(rst_path, "w") as rst_file:
                    rst_file.write(rst_content)

                print(f"Created {rst_path}")


def update_modules_rst(docs_dir, package_names):
    modules_content = "API Reference\n=============\n\n"

    for package_name in package_names:
        modules_content += (
            f"{package_name.capitalize()}\n{'-' * len(package_name)}\n\n"
        )
        modules_content += ".. toctree::\n   :maxdepth: 1\n\n"

        package_dir = os.path.join(docs_dir, package_name)
        for file in os.listdir(package_dir):
            if file.endswith(".rst"):
                modules_content += (
                    f"   {package_name}/{os.path.splitext(file)[0]}\n"
                )

        modules_content += "\n"

    with open(os.path.join(docs_dir, "modules.rst"), "w") as modules_file:
        modules_file.write(modules_content)

    print("Updated modules.rst")


if __name__ == "__main__":
    # Set your directories here
    src_dir = "../src"
    docs_dir = "."
    package_names = ["tricycle", "tricycle_datasets"]

    for package_name in package_names:
        create_rst_files(
            os.path.join(src_dir, package_name), docs_dir, package_name
        )

    update_modules_rst(docs_dir, package_names)
