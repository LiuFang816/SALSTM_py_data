import serial
import struct
import time

_SAMPLE_RATE = 250.0  # Hz
_START_BYTE = bytes(0xA0)  # start of data packet
_END_BYTE = bytes(0xC0)  # end of data packet

_ADS1299_Vref = 4.5  #reference voltage for ADC in ADS1299.  set by its hardware
_ADS1299_gain = 24.0  #assumed gain setting for ADS1299.  set by its Arduino code
_scale_fac_uVolts_per_count = _ADS1299_Vref/(pow(2,23)-1)/(_ADS1299_gain*1000000.)


        
class OpenBCIBoard(object):
  """

  Handle a connection to an OpenBCI board.

  Args:
    port: The port to connect to.
    baud: The baud of the serial connection.

  """

  def __init__(self, port='/dev/tty.usbserial-DN0094CZ', baud=115200, filter_data=True):
       
    self.ser = serial.Serial(port, baud)
    print("Serial established...")

    #Initialize 32-bit board, doesn't affect 8bit board
    self.ser.write('v')

    #wait for device to be ready 
    time.sleep(1)
    self.print_incoming_text()

    self.streaming = False
    self.filtering_data = filter_data
    self.channels = 8
    self.read_state = 0

  def printBytesIn(self):
    #DEBBUGING: Prints individual incoming bytes 
    if not self.streaming:
      self.ser.write('b')
      self.streaming = True
    while self.streaming:
      print(struct.unpack('B',self.ser.read())[0])

  def start(self, callback_functions):
    """
    Start handling streaming data from the board. Call a provided callback
    for every single sample that is processed.

    :param callback_functions: callback functions that will receive a single argument of the
          OpenBCISample object captured.
    """
    if not self.streaming:
      self.ser.write('b')
      self.streaming = True

    while self.streaming:
      sample = self._read_serial_binary()
      for (metric, callback_function) in callback_functions.items():
        callback_function(sample)


  def stop(self):
    """
    Turn streaming off without disconnecting from the board

    """
    self.streaming = False

  def disconnect(self):
    self.ser.close()
    self.streaming = False



  def print_incoming_text(self):
    
    """ 
    When starting the connection, print all the debug data until 
    we get to a line with the end sequence '$$$'.
    """

    #Wait for device to send data
    time.sleep(0.5)
    if self.ser.inWaiting():
      print("-------------------")
      line = ''
      c = ''
     #Look for end sequence $$$
      while '$$$' not in line:
        c = self.ser.read()
        line += c   
      print(line);
      print("-------------------\n")


  def enable_filters(self):
    """
    Adds a filter at 60hz to cancel out ambient electrical noise.
    """
    self.ser.write('f')
    self.filtering_data = True

  def disable_filters(self):
    self.ser.write('g')
    self.filtering_data = False

  def warn(self, text):
    print("Warning: {0}".format(text))


  def _read_serial_binary(self, max_bytes_to_skip=3000):
    """
    Parses incoming data packet into OpenBCISample.
    Incoming Packet Structure:
    Start Byte(1)|Sample ID(1)|Channel Data(24)|Aux Data(6)|End Byte(1)
    0xA0|0-255|8, 3-byte signed ints|3 2-byte signed ints|0xC0
    """
    def read(n):
      b = self.ser.read(n)
      # print bytes(b)
      return b

    for rep in xrange(max_bytes_to_skip):

      #Looking for start and save id when found
      if self.read_state == 0:
        b = read(1)
        if not b:
          if not self.ser.inWaiting():
              self.warn('Device appears to be stalled. Restarting...')
              self.ser.write('b\n')  # restart if it's stopped...
              time.sleep(.100)
              continue
        if bytes(struct.unpack('B', b)[0]) == _START_BYTE:
          if(rep != 0):
            self.warn('Skipped %d bytes before start found' %(rep))
          packet_id = struct.unpack('B', read(1))[0] #packet id goes from 0-255
          
          self.read_state = 1

      elif self.read_state == 1:
        channel_data = []
        for c in xrange(self.channels):

          #3 byte ints
          literal_read = read(3)

          unpacked = struct.unpack('3B', literal_read)

          #3byte int in 2s compliment
          if (unpacked[0] >= 127): 
            pre_fix = '\xFF'
          else:
            pre_fix = '\x00'
          

          literal_read = pre_fix + literal_read; 

          #unpack little endian(>) signed integer(i)
          #also makes unpacking platform independent
          myInt = struct.unpack('>i', literal_read)

          channel_data.append(myInt[0]*_scale_fac_uVolts_per_count)
        
        self.read_state = 2


      elif self.read_state == 2:
        aux_data = []
        for a in xrange(3):

          #short(h) 
          acc = struct.unpack('h', read(2))[0]
          aux_data.append(acc)
    
        self.read_state = 3;

      elif self.read_state == 3:
        val = bytes(struct.unpack('B', read(1))[0])
        if (val == _END_BYTE):
          sample = OpenBCISample(packet_id, channel_data, aux_data)
          self.read_state = 0 #read next packet
          return sample
        else:
          self.warn("Warning: Unexpected END_BYTE found <%s> instead of <%s>,\
            discarted packet with id <%d>" 
            %(val, _END_BYTE, packet_id))

  def test_signal(self, signal):
    if signal == 0:
      self.ser.write('0')
      self.warn("Connecting all pins to ground")
    elif signal == 1:
      self.ser.write('p')
      self.warn("Connecting all pins to Vcc")
    elif signal == 2:
      self.ser.write('-')
      self.warn("Connecting pins to low frequency 1x amp signal")
    elif signal == 3:
      self.ser.write('=')
      self.warn("Connecting pins to high frequency 1x amp signal")
    elif signal == 4:
      self.ser.write('[')
      self.warn("Connecting pins to low frequency 2x amp signal")
    elif signal == 5:
      self.ser.write(']')
      self.warn("Connecting pins to high frequency 2x amp signal")
    else:
      self.warn("%s is not a known test signal. Valid signals go from 0-5" %(signal))

  def set_channel(self, channel, toggle_position):
    #Commands to set toggle to on position
    if toggle_position == 1: 
      if channel is 1:
        self.ser.write('!')
      if channel is 2:
        self.ser.write('@')
      if channel is 3:
        self.ser.write('#')
      if channel is 4:
        self.ser.write('$')
      if channel is 5:
        self.ser.write('%')
      if channel is 6:
        self.ser.write('^')
      if channel is 7:
        self.ser.write('&')
      if channel is 8:
        self.ser.write('*')
    #Commands to set toggle to off position
    elif toggle_position == 0: 
      if channel is 1:
        self.ser.write('1')
      if channel is 2:
        self.ser.write('2')
      if channel is 3:
        self.ser.write('3')
      if channel is 4:
        self.ser.write('4')
      if channel is 5:
        self.ser.write('5')
      if channel is 6:
        self.ser.write('6')
      if channel is 7:
        self.ser.write('7')
      if channel is 8:
        self.ser.write('8')


class OpenBCISample(object):
  """Object encapulsating a single sample from the OpenBCI board."""
  def __init__(self, packet_id, channel_data, aux_data):
    self.id = packet_id
    self.channel_data = channel_data
    self.aux_data = aux_data
    
