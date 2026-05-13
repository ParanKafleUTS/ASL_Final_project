import pkg_resources
import sys

def list_installed_packages():
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print("\n" + "=" * 50)
    print(f"{'Package':<35} {'Version':<20}")
    print("=" * 50)

    packages = sorted(pkg_resources.working_set, key=lambda p: p.project_name.lower())

    for package in packages:
        print(f"{package.project_name:<35} {package.version:<20}")

    print("=" * 50)
    print(f"\nTotal packages installed: {len(packages)}")

if __name__ == "__main__":
    list_installed_packages()
