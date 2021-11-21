from struct import unpack_from as unpack, calcsize

from .models import Vox, Size, Voxel, Color, Model, Material

class ParsingException(Exception):
    pass


def bit(val, offset):
    mask = 1 << offset
    return val & mask

class Chunk(object):
    def __init__(self, chunk_id, content=None, chunks=None):
        self.id = chunk_id
        self.content = content or b''
        self.chunks = chunks or []
        self.material = None

        if chunk_id == b'MAIN':
            if len(self.content):
                raise ParsingException('Non-empty content for main chunk')
        elif chunk_id == b'PACK':
            self.models = unpack('i', content)[0]
        elif chunk_id == b'SIZE':
            self.size = Size(*unpack('iii', content))
        elif chunk_id == b'XYZI':
            n = unpack('i', content)[0]
            print('xyzi block with %d voxels (len %d)', n, len(content))
            self.voxels = []
            self.voxels = [Voxel(*unpack('BBBB', content, 4 + 4 * i)) for i in range(n)]
        elif chunk_id == b'RGBA':
            self.palette = [Color(*unpack('BBBB', content, 4 * i)) for i in range(255)]
            # Docs say:  color [0-254] are mapped to palette index [1-255]
            # hmm
            # self.palette = [ Color(0,0,0,0) ] + [ Color(*unpack('BBBB', content, 4*i)) for i in range(255) ]
        elif chunk_id == b'MATL':
            # print(content)
            _id, _type = unpack('ii', content)

            self.material = Material(_id, _type, content[:4], content[4:8], content[8:])

        else:
            # raise ParsingException('Unknown chunk type: %s'%self.id)
            print(f"Unknown chunk type {chunk_id}")
            pass


class VoxParser(object):

    def __init__(self, filename):
        print("test2")
        with open(filename, 'rb') as f:
            self.content = f.read()

        self.offset = 0

    def unpack(self, fmt):
        r = unpack(fmt, self.content, self.offset)
        self.offset += calcsize(fmt)
        return r

    def _parse_chunk(self):

        _id, n, m = self.unpack('4sii')

        print(f"Found chunk id {_id} / len {n} / children {m}")

        content = self.unpack('%ds' % n)[0]

        start = self.offset
        chunks = []
        while self.offset < start + m:
            chunks.append(self._parse_chunk())

        return Chunk(_id, content, chunks)

    def parse(self):

        print("custom version")

        header, version = self.unpack('4si')

        if header != b'VOX ':
            raise ParsingException("This doesn't look like a vox file to me")

        if version != 150:
            raise ParsingException("Unknown vox version: %s expected 150" % version)

        main = self._parse_chunk()

        if main.id != b'MAIN':
            raise ParsingException("Missing MAIN Chunk")

        chunks = list(reversed(main.chunks))
        if chunks[-1].id == b'PACK':
            models = chunks.pop().models
        else:
            models = 1

        models = []
        for i,c in enumerate(chunks):
            if c.id == b'XYZI':
                assert chunks[i+1].id == b'SIZE'
                print(chunks[i+1].id)
                models.append(self._parse_model(chunks[i+1], c))

        print(f"found {len(models)} models")

        palette = [chunk.palette for chunk in chunks if chunk.id == b'RGBA'][0]
        materials = [chunk.material for chunk in chunks if chunk.id == b'MATL']

        remnants = []
        for c in chunks:
            if c.id not in [b'XYZI', b'SIZE', b'RGBA', b'MATL']:
                remnants.append(c)

        return Vox(models, palette, materials, remnants)

    def _parse_model(self, size, xyzi):
        if size.id != b'SIZE':
            raise ParsingException('Expected SIZE chunk, got %s', size.id)
        if xyzi.id != b'XYZI':
            raise ParsingException('Expected XYZI chunk, got %s', xyzi.id)

        return Model(size.size, xyzi.voxels)


if __name__ == '__main__':
    import sys
    VoxParser(sys.argv[1]).parse()
