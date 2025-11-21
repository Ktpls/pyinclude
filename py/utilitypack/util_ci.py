import pathlib
import netmiko


def SwitchAnchor(path, anchor_old, anchor_new):
    return (
        pathlib.Path(anchor_new)
        / pathlib.Path(path).relative_to(pathlib.Path(anchor_old))
    ).as_posix()


def su_root(
    connection: netmiko.BaseConnection,
    pw: str,
    user: str = "root",
    confirm: bool = False,
):
    output = connection.send_command_timing(f"su {user}")
    assert "Password:" in output or "密码:" in output
    output = connection.send_command_timing(pw)
    # run some command to make sure we are root
    if confirm:
        whoami_output = connection.send_command_timing("whoami")
        assert user in whoami_output, "specified user is not switched to"
