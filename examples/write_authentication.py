import os
import logging
import pacsys
from pacsys import KerberosAuth

os.environ["PACSYS_DPM_TRACE"] = "1"
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# This example shows traces of two possible authentication types for writing via DPM/HTTP.

# First is based on console class of your account
# (see ACL `user_info/console_classes`` and `show <>/setting_protection` command)
with pacsys.dpm(auth=KerberosAuth()) as b:
    print("principal:", b.principal)
    print("no role:", b.write("Z:ACLTST", 45.0))

# Second is based on role-based access control check
with pacsys.dpm(auth=KerberosAuth(), role="testing") as b:
    print("with role:", b.write("Z:ACLTST", 45.0))
