#!/usr/bin/env python3
"""Delete the FABRIC slice. Run this after every experiment to free allocation.

Leaving a slice idle wastes the project's FABRIC allocation, so teardown is a
first-class step, not an afterthought. Safe to run even if the slice does not
exist (it just reports that there's nothing to delete).
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description="Delete the RLSA FABRIC slice.")
    ap.add_argument("--slice-name", default="rlsa-fabric")
    a = ap.parse_args()

    from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager

    fablib = fablib_manager()
    try:
        slice = fablib.get_slice(a.slice_name)
    except Exception as e:
        print(f"No slice '{a.slice_name}' to delete ({e}).")
        return
    slice.delete()
    print(f"Deleted slice '{a.slice_name}'.")


if __name__ == "__main__":
    main()
