__author__ = 'ykhadilkar'

from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy.sql import select, text
from sqlalchemy import func
import ckan.model as model

cached_tables = {}

def get_organization_title(id):
    query = "select id, title from package where package.id = '"+id+"'"
    connection = model.Session.connection()
    res = connection.execute(query).fetchone()
    return res._row[1]

def get_parent_organizations(c):
    items = {}

    if not c.userobj:
        return items

    if c.userobj.sysadmin:
        query = "select package_id, title from package_extra , package " \
                "where package_extra.key = 'is_parent' and package_extra.value = 'true' " \
                "and package.id = package_extra.package_id and package.state = 'active' "
    else:
        userGroupsIds = c.userobj.get_group_ids()
        ids = []

        for id in userGroupsIds:
            ids.append(id.encode('ascii','ignore'))

        # Ugly hack - If user has access to only one organization then SQL query blows up because IN statement ends up with
        # dangling comma at the end. Adding dumy id should fix that.
        if(len(ids) == 0):
            ids.append("null")
            ids.append("dummy-id")
        elif(len(ids) == 1):
            ids.append("dummy-id")
        query = "select package_id, title from package_extra , package " \
                "where package_extra.key = 'is_parent' and package_extra.value = 'true' " \
                "and package.id = package_extra.package_id and package.state = 'active' " \
                "and package.owner_org in "+str(tuple(ids))

    connection = model.Session.connection()
    res = connection.execute(query).fetchall()
    for result in res:
        items[result._row[0]] = result._row[1]

    return items