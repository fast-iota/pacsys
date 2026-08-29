"""Query account and device metadata through ACL over SSH."""

import pacsys

device_name_list = ["M:OUTTMP"]
user = ["toor"]

with pacsys.SSHClient(["clx66.fnal.gov"]) as ssh:
    # Get your account info
    output = ssh.acl(f"user_info/exact {user}")

    # Check how devices are setup, and what SSDN you need for FTP/SNP
    for d in device_name_list:
        property_list = ssh.acl(f"show {d}/ssdns")
        # PRREAD SSDN = 0000/0042/213F/0000 MUONFE
        # PRANAB SSDN = 0000/0042/213F/0000 MUONFE

        console_classes_allowed = ssh.acl(f"show {d}/setting_protection")
        # MCR            Enabled   FSWriteOk      Enabled
        # RemoteMCR      Enabled   Development    Disabled
        # CHL            Enabled   Collider       Enabled
        # MCRCrewChief   Enabled   Linac          Enabled
        # ASTA           Disabled  Booster        Enabled
        # CDF            Enabled   MainInjector   Enabled
        # WebUser        Enabled   Switchyard     Enabled
        # Minimal        Enabled   Tevatron       Enabled
        # Operations     Enabled   NuMI           Disabled
        # KTev           Disabled  BNB            Disabled
        # AccelPrgmmer   Enabled   Meson          Disabled
        # RF             Enabled   Frig           Disabled
        # NOvA           Disabled  Accel R&D      Enabled
        # CentralOther   Enabled   RemoteSet      Disabled
        # CentralServ    Enabled   Manager        Disabled
        # Muon           Enabled
