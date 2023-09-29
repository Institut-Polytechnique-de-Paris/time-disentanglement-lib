"""
TS Metadata reader and parser. It iterates through each TS packet and decodes the header. It also provides a
way to skip to any packet non-linearly, and supports iterating backwards.

It is useful for basic TS metadata analysis scripting, or could be used as the basis of a more complex TS metadata
analysis tool.

Note: Does not decode or de-multiplex a/v.

Also has command line interface to measure the duration of ts files based on pts or dts. Run as main for cli.

Simple Example Usage:

    file = "C:/videofiles/tsfile.ts"
    ts = TSRead(file)
    for x in ts:
        print x.decodedpts, x.decodeddts
        
        
Note 2: The code is not beautiful, but it works. I plan to pretty it up in the future.

Copyright Joshua Banton
"""

import argparse
import bitstring
import sys
import argparse
import os

class TSRead():
    """
    Class to read TS files.
    """
    def __init__(self, tsfile):
        self.tsfile = tsfile
        self.tsopen = open(tsfile, 'rb')
        self.pos = -188
        self.bytes = self.tsopen.read(188)
        self.bits = bitstring.ConstBitStream(bytes=self.bytes, length=1504)
        self.packetnum = 0
        self.filesize = os.path.getsize(tsfile)
        self.totalpackets = self.filesize/188
        if not (self.filesize % 188) == 0:
            self.complete = False
        else:
            self.complete = True
    
    def next(self):
        """reads the ts header and returns all the header values"""
        self.pos += 188
        if self.pos == self.filesize:
            raise StopIteration
        if not self.pos == self.filesize:
            self.tsopen.seek(self.pos)
            self.bytes = self.tsopen.read(188)
            self.bits = bitstring.ConstBitStream(bytes=self.bytes, length=1504)
            self.packetnum += 1
        tsdata = self.decodets()
         
        return tsdata    
        
    def back(self):
        """reads the ts header and returns all the header values"""
        self.pos -= 188
        if self.pos == 0:
            raise StopIteration
        if not self.pos == self.filesize:
            self.tsopen.seek(self.pos)
            self.bytes = self.tsopen.read(188)
            self.bits = bitstring.ConstBitStream(bytes=self.bytes, length=1504)
            self.packetnum -= 1
        tsdata = self.decodets()
         
        return tsdata
        
    def __iter__(self):
        return self
        
    def last(self):
        self.pos = self.filesize - 188
        self.tsopen.seek(self.pos)
        self.bytes = self.tsopen.read(188)
        self.bits = bitstring.ConstBitStream(bytes=self.bytes, length=1504)
        self.packetnum = self.totalpackets
        tsdata = self.decodets()
        return tsdata    
        
    def first(self):
        self.pos = -188
        self.packetnum = 0
        tsdata = self.next()
        return tsdata
        
    def goto(self, packetnum):
        self.pos = (packetnum * 188) - 376
        self.packetnum = packetnum - 1
        tsdata = self.next()
        return tsdata
        
    def decodets(self):
        adaptation_size = False
        av = False
        adapt = False
        pestype = False
        ptspresent = False
        dtspresent = False
        decodedpts = False
        decodeddts = False
        pcr = False
        opcr = False
        discontinuity = False
        random = False
        espriority = False
        pcrpresent = False
        opcrpresent = False
        splicingpoint = False
        transportprivate = False
        adaptation_ext = False
        packsize = 188
        sync = self.bits.read(8).hex
        if sync == '0x47':
            tei = self.bits.read(1).uint
            pusi = self.bits.read(1).uint
            transportpri = self.bits.read(1).uint
            pid = self.bits.read(13).uint
            packet = self.bits.read((packsize-3)*8)
            scramblecontrol = packet.read(2).uint
            adapt = packet.read(2).uint
            concounter = packet.read(4).uint
            if adapt == 3:
                adaptation_size = packet.read(8).uint
                discontinuity = packet.read(1).uint
                random = packet.read(1).uint
                espriority = packet.read(1).uint
                pcrpresent = packet.read(1).uint
                opcrpresent = packet.read(1).uint
                splicingpoint = packet.read(1).uint
                transportprivate = packet.read(1).uint
                adaptation_ext = packet.read(1).uint
                restofadapt = (adaptation_size+3) - 1
                if pcrpresent == 1:
                    pcr = packet.read(48)
                    restofadapt -=  6
                if opcrpresent == 1:
                    opcr = packet.read(48)
                    restofadapt -=  6
                packet.pos += (restofadapt-3) * 8
                if ((packet.len - packet.pos)/8) > 5:
                    pesync = packet.read(24).hex
                    if pesync == ('0x000001'):
                        pestype = packet.read(8).uint
                        if pestype > 223 and pestype < 240:
                            av = 'video'
                        if pestype < 223 and pestype > 191:
                            av = 'audio'
                        packet.pos += (3*8)
                        ptspresent = packet.read(1).uint
                        dtspresent = packet.read(1).uint
                        if ptspresent:
                            packet.pos += (14)
                            pts = packet.read(40)
                            pts.pos = 4
                            firstpartpts = pts.read(3)
                            pts.pos += 1
                            secondpartpts = pts.read(15)
                            pts.pos += 1
                            thirdpartpts = pts.read(15)
                            decodedpts = bitstring.ConstBitArray().join([firstpartpts.bin, secondpartpts.bin, thirdpartpts.bin]).uint
                        if dtspresent:
                            dts = packet.read(40)
                            dts.pos = 4
                            firstpartdts = dts.read(3)
                            dts.pos += 1
                            secondpartdts = dts.read(15)
                            dts.pos += 1
                            thirdpartdts = dts.read(15)
                            decodeddts = bitstring.ConstBitArray().join([firstpartdts.bin, secondpartdts.bin, thirdpartdts.bin]).uint
            elif adapt == 2:
                #if adapt is 2 the packet is only an adaptation field
                adaptation_size = packet.read(8).uint
                discontinuity = packet.read(1).uint
                random = packet.read(1).uint
                espriority = packet.read(1).uint
                pcrpresent = packet.read(1).uint
                opcrpresent = packet.read(1).uint
                splicingpoint = packet.read(1).uint
                transportprivate = packet.read(1).uint
                adaptation_ext = packet.read(1).uint
                restofadapt = (adaptation_size+3) - 1
                if pcrpresent == 1:
                    pcr = packet.read(48)
                    restofadapt -=  6
                if opcrpresent == 1:
                    opcr = packet.read(48)
                    restofadapt -=  6
            elif adapt == 1:
                pesync = packet.read(24).hex
                if pesync == ('0x000001'):
                    pestype = packet.read(8).uint
                    if pestype > 223 and pestype < 240:
                        av = 'video'
                    if pestype < 223 and pestype > 191:
                        av = 'audio'
                    packet.pos += 24
                    ptspresent = packet.read(1).uint
                    dtspresent = packet.read(1).uint
                    if ptspresent:
                        packet.pos += 22
                        pts = packet.read(32)
                        firstpart = pts.read(15)
                        pts.pos += 1
                        secondpart = pts.read(15)
                        decodedpts = bitstring.ConstBitArray().join([firstpart.bin, secondpart.bin]).uint
                    if dtspresent:
                            dts = packet.read(40)
                            dts.pos = 4
                            firstpartdts = dts.read(3)
                            dts.pos += 1
                            secondpartdts = dts.read(15)
                            dts.pos += 1
                            thirdpartdts = dts.read(15)
                            decodeddts = bitstring.ConstBitArray().join([firstpartdts.bin, secondpartdts.bin, thirdpartdts.bin]).uint
                
        else:
            return 'found badness?', sync.hex
        
        tsobj = TSPacket(sync, tei, transportpri, pusi, pid, scramblecontrol, adapt, concounter, adaptation_size, discontinuity, \
        random, espriority, pcrpresent, opcrpresent, splicingpoint, transportprivate, adaptation_ext, pcr, opcr, pestype, \
        ptspresent, dtspresent, decodedpts, decodeddts, av)
        
        return tsobj
  
class TSPacket():
    """
    Class to read TS packets.
    """
    def __init__(self, sync, tei, transportpri, pusi, pid, scramblecontrol, adapt, concounter, adaptation_size, discontinuity, \
    random, espriority, pcrpresent, opcrpresent, splicingpoint, transportprivate, adaptation_ext, pcr, opcr, pestype, \
    ptspresent, dtspresent, decodedpts, decodeddts, av):
        self.sync = sync
        self.transportpri = transportpri
        self.pusi = pusi
        self.pid = pid
        self.scramblecontrol = scramblecontrol
        self.adapt = adapt
        self.concounter = concounter
        self.adaptation_size = adaptation_size
        self.discontinuity = discontinuity
        self.random = random
        self.espriority = espriority
        self.pcrpresent = pcrpresent
        self.opcrpresent = opcrpresent
        self.splicingpoint = splicingpoint
        self.transportprivate = transportprivate
        self.adaptation_ext = adaptation_ext
        self.pcr = pcr
        self.opcr = opcr
        self.pestype = pestype
        self.ptspresent = ptspresent
        self.dtspresent = dtspresent
        self.decodedpts = decodedpts
        self.decodeddts = decodeddts
        self.av = av
        
def get_duration_format(file, format, type):
    run = True
    ts = TSRead(file)
    for x in ts:
        if type == 'pts':
            data = x
            first = data.decodedpts
            if x.ptspresent and x.av == format:
                break
        else:
            data = x
            first = data.decodeddts
            if x.dtspresent and x.av == format:
                break
    firstdata = ts.last()
    if type == 'pts':
        if firstdata.ptspresent and firstdata.av == format:
            run = False
    else:
        if firstdata.dtspresent and firstdata.av == format:
            run = False
    while run:
        if type == 'pts':
            data = ts.back()
            last = data.decodedpts
            if data.ptspresent and data.av == format:
                break
        else:
            data = ts.back()
            last = data.decodeddts
            if data.dtspresent and data.av == format:
                break
    duration = (last-first)/90000.0
    return duration

def get_duration_pid(file, pid, type):
    run = True
    ts = TSRead(file)
    for x in ts:
        if type == 'pts':
            data = x
            first = data.decodedpts
            if x.ptspresent and x.pid == int(pid):
                break
        else:
            data = x
            first = data.decodeddts
            if x.dtspresent and x.pid == int(pid):
                break
    firstdata = ts.last()
    if type == 'pts':
        if firstdata.ptspresent and firstdata.pid == int(pid):
            run = False
    else:
        if firstdata.dtspresent and firstdata.pid == int(pid):
            run = False
    while run:
        if type == 'pts':
            data = ts.back()
            last = data.decodedpts
            if data.ptspresent and data.pid == int(pid):
                break
        else:
            data = ts.back()
            last = data.decodeddts
            if data.dtspresent and data.pid == int(pid):
                break
    duration = (last-first)/90000.0
    return duration

def batchfolder(dir):
    os.chdir(dir)
    files = os.listdir(os.path.abspath(dir))
    ts_files = []
    for file in files:
        if '.ts' in file:
            ts_files.append(os.path.abspath(file))
    return ts_files
    
if __name__ == "__main__":
    """Command line interface to measure the duration of ts files based on pts or dts"""
    parser = argparse.ArgumentParser(description='Measures ts durations by pts or dts of the audio or video')
    parser.add_argument('-i', '--input', nargs='?', default=None, help=\
                            "input ts file path to calculate duration of")
    parser.add_argument('-b', '--batch', nargs='?', const=True, default=False, help=\
    "folder to batch process")
    parser.add_argument('-f', '--format', nargs='?', default='video', help=\
    "measure video or audio | Note: Stream type must be labeled correctly")
    parser.add_argument('-p', '--pid', nargs='?', default=False, help=\
    "measure pid")
    parser.add_argument('-t', '--type', nargs='?', default='pts', help=\
    "measure duration by pts or dts")
    args = parser.parse_args()
    
    file = args.input
    batch = args.batch
    
    if file:
        if not args.pid:
            print get_duration_format(file, args.format, args.type)
        else:
            print get_duration_pid(file, args.pid, args.type)
    elif batch:
        files = batchfolder(batch)
        for x in files:
            print get_duration(file, format, type)
    else:
        print 'need an input file'