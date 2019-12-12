# -*- coding: utf-8 -*-
# Copyright 2016 OpenMarket Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import resource
import logging


logger = logging.getLogger("synapse.app.homeserver")


def change_resource_limit(soft_file_no):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        if not soft_file_no:
            soft_file_no = hard

        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_file_no, hard))
        logger.info("Set file limit to: %d", soft_file_no)

        resource.setrlimit(
            resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
    except (ValueError, resource.error) as e:
        logger.warn("Failed to set file or core limit: %s", e)
