import struct
import collections

from bfnet import util


class _MetaPacket(type):
    """
    This Metaclass exists only to prepare an ordered dict.
    """

    @classmethod
    def __prepare__(mcs, name, bases):
        return collections.OrderedDict()


class BasePacket(object, metaclass=_MetaPacket):
    """
    A BasePacket is the base class for all Packet types.

    This just creates a few stub methods.
    """

    # Define a default id.
    # Packet IDs are ALWAYS unsigned, so this will never meet.
    id = -1

    # Define a default endianness.
    # This is ">" for network endianness by default.
    _endianness = ">"

    def __init__(self, pbf):
        """
        Default init method.
        """
        self.butterfly = pbf

    def on_creation(self):
        """
        Called just after your packet object is created.
        """
        pass

    def create(self, data: bytes):
        """
        Create a new Packet.
        :param data: The data to use.
        :return: If the creation succeeded or not.
        """
        self.on_creation()

    def autopack(self) -> bytes:
        """
        Attempt to autopack your data correctly.

        This does two things:
            - Scan your class dictionary for all non-function and struct-packable
            items.
            - Infer their struct format type, build a format string, then pack them.
        :return: The packed bytes data.
        """
        # Get the variables.
        to_fmt = []
        v = vars(self)
        for variable, val in v.items():
            # Get a list of valid types.
            if type(val) not in [bytes, str, int, float]:
                self.butterfly.logger.debug("Found un-packable type: {}, skipping".format(type(val)))
            elif variable.startswith("_"):
                self.butterfly.logger.debug("Found private variable {}, skipping".format(variable))
            elif variable.lower() == "id":
                self.butterfly.logger.debug("Skipping ID variable")
            else:
                to_fmt.append(val)
        packed = util.auto_infer_struct_pack(*to_fmt, pack=True)
        return packed


class Packet(BasePacket):
    """
    A standard Packet type.

    This extends from BasePacket, and adds useful details that you'll want to use.
    """

    def __init__(self, pbf):
        """
        Create a new Packet type.
        :return:
        """
        super().__init__(pbf)
        self._original_data = {}

    def create(self, data: dict) -> bool:
        """
        Create a new Packet.
        :param data: The data to use.
            This data should have the PacketButterfly header stripped.
        :return: A boolean, True if we need no more processing, and False if we process ourself.
        """
        self._original_data = data
        self.unpack(data)
        return True

    def unpack(self, data: dict) -> bool:
        """
        Unpack the data for the packet.
        :return: A boolean, if it was unpacked.
        """
        return True

    def gen(self) -> bytes:
        """
        Generate a new set of data to write to the connection.
        :return:
        """
