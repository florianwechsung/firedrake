"""The :mod:`firedrake_configuration` module records the configuration
with which Firedrake was last installed or updated. It is a separate
package from Firedrake in order to ensure that `firedrake-update` can
always access the configuration, even if the :mod:`.firedrake` module
itself is broken."""

import json
import os

# Attempt to read configuration from file.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "configuration.json"), "r") as f:
        _config = json.load(f)
except IOError:
    _config = None


def write_config(config):
    """Output the configuration provided to disk. This routine should
    normally only be called by `firedrake-install` or
    `firedrake-update`."""

    # Create the json as a separate step
    json_output = json.dumps(config)
    with open(os.path.join(os.path.dirname(__file__),
                           "configuration.json"), "w") as f:
        f.write(json_output)


def get_config():
    """Return the current configuration dictionary"""
    return _config


def get_config_json():
    """Return a json serialisation of the current configuration. This
    could be output by a Firedrake application to assist in the
    reproduction of results."""
    return json.dumps(_config)


def setup_cache_dirs():
    config = get_config()
    if "PYOP2_CACHE_DIR" not in os.environ:
        os.environ["PYOP2_CACHE_DIR"] = os.path.join(config["options"]["cache_dir"], "pyop2")
    if 'FIREDRAKE_TSFC_KERNEL_CACHE_DIR' not in os.environ:
        os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = os.path.join(config["options"]["cache_dir"], "tsfc")
