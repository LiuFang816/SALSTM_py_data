import binascii
import rpcBase
import struct
import uuid

from dcerpc import MSRPCHeader, MSRPCBindAck
from structure import Structure

uuidNDR32 = uuid.UUID('8a885d04-1ceb-11c9-9fe8-08002b104860')
uuidNDR64 = uuid.UUID('71710533-beba-4937-8319-b5dbef9ccc36')
uuidTime = uuid.UUID('6cb71c2c-9812-4540-0300-000000000000')
uuidEmpty = uuid.UUID('00000000-0000-0000-0000-000000000000')

class CtxItem(Structure):
	structure = (
		('ContextID',          '<H=0'),
		('TransItems',         'B=0'),
		('Pad',                'B=0'),
		('AbstractSyntaxUUID', '16s=""'),
		('AbstractSyntaxVer',  '<I=0'),
		('TransferSyntaxUUID', '16s=""'),
		('TransferSyntaxVer',  '<I=0'),
	)

	def ts(self):
		return uuid.UUID(bytes_le=self['TransferSyntaxUUID'])

class CtxItemResult(Structure):
	structure = (
		('Result',             '<H=0'),
		('Reason',             '<H=0'),
		('TransferSyntaxUUID', '16s=""'),
		('TransferSyntaxVer',  '<I=0'),
	)

	def __init__(self, result, reason, tsUUID, tsVer):
		Structure.__init__(self)
		self['Result'] = result
		self['Reason'] = reason
		self['TransferSyntaxUUID'] = tsUUID.bytes_le
		self['TransferSyntaxVer'] = tsVer

class MSRPCBind(Structure):
	class CtxItemArray:
		def __init__(self, data):
			self.data = data

		def __len__(self):
			return len(self.data)

		def __str__(self):
			return self.data

		def __getitem__(self, i):
			return CtxItem(self.data[(len(CtxItem()) * i):])

	_CTX_ITEM_LEN = len(CtxItem())

	structure = (
			('max_tfrag',   '<H=4280'),
			('max_rfrag',   '<H=4280'),
			('assoc_group', '<L=0'),
			('ctx_num',     'B=0'),
			('Reserved',    'B=0'),
			('Reserved2',   '<H=0'),
			('_ctx_items',  '_-ctx_items', 'self["ctx_num"]*self._CTX_ITEM_LEN'),
			('ctx_items',   ':', CtxItemArray),
	)

class handler(rpcBase.rpcBase):
	def parseRequest(self):
		request = MSRPCHeader(self.data)

		if self.config['debug']:
			print "RPC Bind Request Bytes:", binascii.b2a_hex(self.data)
			print "RPC Bind Request:", request.dump(), MSRPCBind(request['pduData']).dump()

		return request

	def generateResponse(self):
		response = MSRPCBindAck()
		request = self.requestData
		bind = MSRPCBind(request['pduData'])

		response['ver_major'] = request['ver_major']
		response['ver_minor'] = request['ver_minor']
		response['type'] = self.packetType['bindAck']
		response['flags'] = self.packetFlags['firstFrag'] | self.packetFlags['lastFrag'] | self.packetFlags['multiplex']
		response['representation'] = request['representation']
		response['frag_len'] = 36 + bind['ctx_num'] * 24
		response['auth_len'] = request['auth_len']
		response['call_id'] = request['call_id']

		response['max_tfrag'] = bind['max_tfrag']
		response['max_rfrag'] = bind['max_rfrag']
		response['assoc_group'] = 0x1063bf3f

		port = str(self.config['port'])
		response['SecondaryAddrLen'] = len(port) + 1
		response['SecondaryAddr'] = port
		pad = (4-((response["SecondaryAddrLen"]+MSRPCBindAck._SIZE) % 4))%4
		response['Pad'] = '\0' * pad
		response['ctx_num'] = bind['ctx_num']

		preparedResponses = {}
		preparedResponses[uuidNDR32] = CtxItemResult(0, 0, uuidNDR32, 2)
		preparedResponses[uuidNDR64] = CtxItemResult(2, 2, uuidEmpty, 0)
		preparedResponses[uuidTime] = CtxItemResult(3, 3, uuidEmpty, 0)

		response['ctx_items'] = ''
		for i in range (0, bind['ctx_num']):
			ts_uuid = bind['ctx_items'][i].ts()
			resp = preparedResponses[ts_uuid]
			response['ctx_items'] += str(resp)

		if self.config['debug']:
			print "RPC Bind Response:", response.dump()
			print "RPC Bind Response Bytes:", binascii.b2a_hex(str(response))

		return response

class bind(rpcBase.rpcBase):
	def generateRequest(self):
		firstCtxItem = CtxItem()
		firstCtxItem['ContextID'] = 0
		firstCtxItem['TransItems'] = 1
		firstCtxItem['Pad'] = 0
		firstCtxItem['AbstractSyntaxUUID'] = uuid.UUID('51c82175-844e-4750-b0d8-ec255555bc06').bytes_le
		firstCtxItem['AbstractSyntaxVer'] = 1
		firstCtxItem['TransferSyntaxUUID'] = uuidNDR32.bytes_le
		firstCtxItem['TransferSyntaxVer'] = 2

		secondCtxItem = CtxItem()
		secondCtxItem['ContextID'] = 1
		secondCtxItem['TransItems'] = 1
		secondCtxItem['Pad'] = 0
		secondCtxItem['AbstractSyntaxUUID'] = uuid.UUID('51c82175-844e-4750-b0d8-ec255555bc06').bytes_le
		secondCtxItem['AbstractSyntaxVer'] = 1
		secondCtxItem['TransferSyntaxUUID'] = uuidTime.bytes_le
		secondCtxItem['TransferSyntaxVer'] = 1

		bind = MSRPCBind()
		bind['max_tfrag'] = 5840
		bind['max_rfrag'] = 5840
		bind['assoc_group'] = 0
		bind['ctx_num'] = 2
		bind['ctx_items'] = bind.CtxItemArray(str(firstCtxItem)+str(secondCtxItem))

		request = MSRPCHeader()
		request['ver_major'] = 5
		request['ver_minor'] = 0
		request['type'] = self.packetType['bindReq']
		request['flags'] = self.packetFlags['firstFrag'] | self.packetFlags['lastFrag'] | self.packetFlags['multiplex']
		request['call_id'] = self.config['call_id']
		request['pduData'] = str(bind)

		if self.config['debug']:
			print "RPC Bind Request:", request.dump(), MSRPCBind(request['pduData']).dump()
			print "RPC Bind Request Bytes:", binascii.b2a_hex(str(request))

		return request

	def parseResponse(self):
		return response

