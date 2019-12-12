class ObjectParsing:
    def get_param(self, blob, key):
        """
        Response from the Virtual Radar service is pretty inconsistent
        depending on the aircraft info available. So this is to help
        tamper that.
        """
        if key in blob:
            return blob[key]
        else:
            return None
