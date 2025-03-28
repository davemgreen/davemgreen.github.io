import sys, os, subprocess, argparse, logging, json, tempfile, multiprocessing, shutil, re

# Try to more extensively check the cost model figures coming out of the cost model, for every operation x type combo.
# Currently it looks at costsize costs, as those are easier to measure.
# Measures codesize from llc with some filtering.
# Can add other costs in the future, they are more difficult to measure correctly.

# Run this to generate data.json
#   python llvm/utils/costmodeltest.py
# and this to serve is to pert 8081, inside a venv with pandas
#   python llvm/utils/costmodeltest.py --servellvm/utils/costmodeltest.py

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format='')
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def run(cmd):
  logging.debug('> ' + cmd)
  cmd = cmd.split()
  return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')

def getcost(path, costkind, print):
  try:
    text = run(f"opt {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} {os.path.join(path, 'costtest.ll')} -passes=print<cost-model> -cost-kind={costkind} -disable-output")
  except subprocess.CalledProcessError as e:
    shutil.copyfile(os.path.join(path, 'costtest.ll'), 'costtest.ll')
    raise
  costpre = 'Cost Model: Found an estimated cost of '
  if print:
    logging.debug(text.strip())
  costs = [x for x in text.split('\n') if 'instruction:   ret ' not in x]
  cost = sum([int(x[len(costpre):len(costpre)+x[len(costpre):].find(' ')]) for x in costs if x.startswith(costpre)])
  return (cost, text.strip())

def getasm(path, extraflags):
  try:
    run(f"llc {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} {extraflags} {os.path.join(path, 'costtest.ll')} -o {os.path.join(path, 'costtest.s')}")
  except subprocess.CalledProcessError as e:
    return ([e.output.decode('utf-8').split('\n')[0]], -1)
  with open(os.path.join(path, "costtest.s")) as f:
    lines = [l.strip() for l in f]
  # This tries to remove .declarations, comments etc
  lines = [l for l in lines if l[0] != '.' and l[0] != '/' and not l.startswith('test:')]
  #logging.debug(lines)

  # TODOD: Improve the filtering to what is invariant, somehow. Or include it in the costs.
  #filteredlines = [l for l in lines if not l.startswith('movi') and not l.startswith('mov\tw') and l != 'ret' and not l.startswith('adrp') and not l.startswith('ldr') and not l.startswith('dup') and not l.startswith('fmov')]
  filteredlines = [l for l in lines if l != 'ret' and not l.startswith('ptrue') and not re.match(r'fmov\sd[0-9], d[0-9]+',l) and not re.match(r'mov\sv[0-9].16b, v[0-9]+.16b', l)]
  logging.debug(filteredlines)
  size = len(filteredlines)
  logging.debug(f"size = {size}")

  return (lines, size)

def checkcosts(llasm):
  logging.debug(llasm)
  with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "costtest.ll"), "w") as f:
      f.write(llasm)

    lines, size = getasm(tmp, '')

    gilines, gisize = getasm(tmp, '-global-isel')

    codesize = getcost(tmp, 'code-size', True)
    thru = getcost(tmp, 'throughput', False)
    lat = getcost(tmp, 'latency', False)
    sizelat = getcost(tmp, 'size-latency', False)

    logging.debug(f"cost = codesize:{codesize[0]} throughput:{thru[0]} lat:{lat[0]} sizelat:{sizelat[0]}")
    return (size, gisize, [codesize, thru, lat, sizelat], llasm, ('\n'.join(lines)).replace('\t', ' '), ('\n'.join(gilines)).replace('\t', ' '))

  # TODOD:
  #if args.checkopted:
  #  run(f"opt {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} costtest.ll -O1 -S -o -")


def generate_const(ty, sameval):
  consts = ['7', '6'] if not ty.isFloat() else ['7.0', '6.0']
  if ty.elts == 1:
    return consts[0]
  if sameval:
    return f"splat ({ty.scalar} {consts[0]})"
  return "<" + ", ".join([f"{ty.scalar} {consts[0]}, {ty.scalar} {consts[1]}" for x in range(ty.elts // 2)]) + '>'
def generate_const0(ty):
  if ty.elts == 1:
    return '0' if not ty.isFloat() else '0.0'
  return "zeroinitializer"
def generate_constm1(ty):
  if ty.elts == 1:
    return '-1'
  return f"splat ({ty.scalar} -1)"

def generate(variant, instr, ty, ty2):
  tystr = ty.str()
  eltstr = ty.scalar
  rettystr = eltstr if instr == 'extractelement' else ty2.str()
  preamble = f"define {rettystr} @test({tystr} %a"
  if variant == 'binop' or variant == 'cmp':
    preamble += f", {tystr} %b"
  elif variant == 'binopsplat' or instr == 'insertelement':
    preamble += f", {eltstr} %bs"
  elif variant == 'triop' and instr == 'select':
    preamble += f", {tystr} %b, {Ty('i1', ty.elts, ty.scalable)} %c"
  elif variant == 'triop':
    preamble += f", {tystr} %b, {tystr} %c"
  elif variant == 'reduce' and (instr == 'reduce.fadd' or instr == 'reduce.fmul'):
    preamble += f", {rettystr} %b"
  if variant == 'vecopvar':
    preamble += f", i32 %c"
  if (variant == 'cmp' or variant == 'cmp0') and instr.startswith('select'):
    preamble += f", {tystr} %d, {tystr} %e"
  preamble += ") {\n"

  setup = ""
  if variant == "binopsplat":
    setup += f"  %i = insertelement {tystr} poison, {eltstr} %bs, i64 0\n"
    setup += f"  %b = shufflevector {tystr} %i, {tystr} poison, <{ty.vecpart()} x i32> zeroinitializer\n"

  instrstr = ""
  b = "%b"
  if "const" in variant:
    b = generate_const(ty, variant == 'binopconstsplat')
  elif variant == 'mvn':
    b = generate_constm1(ty)
  elif variant == 'cmp0':
    b = generate_const0(ty)

  if instr in ['add', 'sub', 'mul', 'sdiv', 'srem', 'udiv', 'urem', 'and', 'or', 'xor', 'shl', 'ashr', 'lshr', 'fadd', 'fsub', 'fmul', 'fdiv', 'frem']:
    instrstr += f"  %r = {instr} {tystr} %a, {b}\n"
  elif instr in ['rotr', 'rotl']:
    instrstr += f"  %r = call {tystr} @llvm.fsh{instr[3]}({tystr} %a, {tystr} %a, {tystr} {b})\n"
  elif instr in ['fneg']:
    instrstr += f"  %r = {instr} {tystr} %a\n"
  elif instr == 'abs' or instr == 'ctlz' or instr == 'cttz':
    instrstr += f"  %r = call {tystr} @llvm.{instr}({tystr} %a, i1 0)\n"
  elif variant == 'unop':
    instrstr += f"  %r = call {tystr} @llvm.{instr}({tystr} %a)\n"
  elif instr in ['select']:
    instrstr += f"  %r = {instr}  {Ty('i1', ty.elts, ty.scalable)} %c, {tystr} %a, {tystr} %b\n"
  elif (variant == 'cmp' or variant == 'cmp0') and instr.startswith('select'):
    instrstr += f"  %c = {instr[6:10]} {instr[10:]} {tystr} %a, {b}\n  %r = select {Ty('i1', ty.elts, ty.scalable)} %c, {tystr} %d, {tystr} %e\n"
  elif variant == 'cmp' or variant == 'cmp0':
    instrstr += f"  %r = {instr[:4]} {instr[4:]} {tystr} %a, {b}\n"
  elif variant == 'triop':
    instrstr += f"  %r = call {tystr} @llvm.{instr}({tystr} %a, {tystr} %b, {tystr} %c)\n"
  elif variant == 'reduce' and (instr == 'reduce.fadd' or instr == 'reduce.fmul'):
    instrstr += f"  %r = call {rettystr} @llvm.vector.{instr}({rettystr} %b, {tystr} %a)\n"
  elif variant == 'reduce':
    instrstr += f"  %r = call {rettystr} @llvm.vector.{instr}({tystr} %a)\n"
  elif instr == 'extractelement':
    idx = '%c' if variant == 'vecopvar' else ('1' if variant == 'vecop1' else '0')
    instrstr += f"  %r = extractelement {tystr} %a, i32 {idx}\n"
  elif instr == 'insertelement':
    idx = '%c' if variant == 'vecopvar' else ('1' if variant == 'vecop1' else '0')
    instrstr += f"  %r = insertelement {tystr} %a, {eltstr} %bs, i32 {idx}\n"
  elif variant.startswith('cast'):
    instrstr += f"  %r = {instr} {tystr} %a to {rettystr}\n"
  else:
    instrstr += f"  %r = call {tystr} @llvm.{instr}({tystr} %a, {tystr} {b})\n"

  return preamble + setup + instrstr + f"  ret {rettystr} %r\n}}"

class Ty:
  def __init__(self, scalar, elts=1, scalable=0):
    self.scalar = scalar
    self.elts = elts
    self.scalable = scalable
  def isFloat(self):
    return self.scalar[0] != 'i'
  def scalarsize(self):
    return int(self.scalar[1:]) if self.scalar[0] == 'i' else fptymap[self.scalar]
  def str(self):
    if self.elts == 1 and not self.scalable:
      return self.scalar
    return f"<{self.vecpart()} x {self.scalar}>"
  def vecpart(self):
    if self.scalable:
      return f"vscale x {self.elts}"
    return f"{self.elts}"
  def __repr__(self):
    return self.str()
fptymap = { 16:'half', 32:'float', 64:'double',
            'half':16, 'float':32, 'double':64 }

def inttypes(highsizes = False):
  # TODO: i128, other type sizes?
  for bits in [8, 16, 32, 64]:
    yield Ty('i'+str(bits))
  for scalable in [0,1]:
    if scalable == 1 and (not args.mattr or 'sve' not in args.mattr):
      continue
    for bits in [8, 16, 32, 64]:
      for s in [2, 4, 8, 16, 32]:
        if not highsizes and s * bits > 256:
          continue
        if s * bits < 64: #TODO: Start looking at smaller sizes once the legal sizes are better. Odd vector sizes
          continue
        yield Ty('i'+str(bits), s, scalable)
def fptypes(highsizes = False):
  # TODO: f128? They are just libcalls
  for bits in [16, 32, 64]:
    yield Ty(fptymap[bits])
  for scalable in [0,1]:
    if scalable == 1 and (not args.mattr or 'sve' not in args.mattr):
      continue
    for bits in [16, 32, 64]:
      for s in [2, 4, 8, 16, 32]:
        if not highsizes and s * bits > 256:
          continue
        if s * bits < 64: # TODO: Start looking at smaller sizes once the legal sizes are better. Odd vector sizes
          continue
        yield Ty(fptymap[bits], s, scalable)

def binop_variants(ty):
  yield ('binop', 0)
  if not ty.scalable:
    yield ('binopconst', 0) #1 if ty.elts == 1 else 2)
  if ty.elts > 1:
    yield ('binopsplat', 0) #1
    yield ('binopconstsplat', 0) #1 if ty.elts == 1 or ty.bits <= 32 else 2)

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['all', 'int', 'fp', 'castint', 'castfp', 'vec'], default='all')
parser.add_argument('-mtriple', default='aarch64')
parser.add_argument('-mattr', default=None)
#parser.add_argument('--checkopted', action='store_true')
args = parser.parse_args()


def do(instr, variant, ty, ty2, extrasize, tyoverride):
  logging.info(f"{variant} {instr} with {ty.str()}")
  (size, gisize, costs, ll, asm, giasm) = checkcosts(generate(variant, instr, ty, ty2))
  tystr = str(ty) if not tyoverride else tyoverride
  if costs[0][0] != size - extrasize:
    logging.warning(f">>> {variant} {instr} with {tystr}  size = {size} vs cost = {costs[0][0]} (expected extrasize={extrasize})")
  return {"instr":instr, "ty":tystr, "variant":variant, "codesize":costs[0][0], "thru":costs[1][0], "lat":costs[2][0], "sizelat":costs[3][0], "size":size, "gisize":gisize, "extrasize":extrasize, "asm":asm, "giasm":giasm, "ll":ll, "costoutput":costs[0][1]}

# Operations are the ones in https://github.com/llvm/llvm-project/issues/115133
#  TODO: load/store, bitcast, getelementptr, phi

if args.type == 'all' or args.type == 'int':
  def enumint():
    # Int Binops
    for instr in ['add', 'sub', 'mul', 'sdiv', 'srem', 'udiv', 'urem', 'and', 'or', 'xor', 'shl', 'ashr', 'lshr', 'smin', 'smax', 'umin', 'umax', 'uadd.sat', 'usub.sat', 'sadd.sat', 'ssub.sat', 'rotr', 'rotl']:
      for ty in inttypes():
        for (variant, extrasize) in binop_variants(ty):
          yield (instr, variant, ty, ty, extrasize, None)

    ## Int unops
    for instr in ['abs', 'bitreverse', 'bswap', 'ctlz', 'cttz', 'ctpop']:
      for ty in inttypes():
        if instr == 'bswap' and ty.scalar == 'i8':
          continue
        yield (instr, 'unop', ty, ty, 0, None)
    for instr in ['xor']:
      for ty in inttypes():
        yield (instr, 'mvn', ty, ty, 0, None)

    # Int triops
    for instr in ['fshl', 'fshr', 'select']:
      for ty in inttypes():
        yield (instr, 'triop', ty, ty, 0, None)

    for op in ['eq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge']:
      for ty in inttypes():
        yield ('icmp'+op, 'cmp', ty, Ty('i1', ty.elts, ty.scalable), 0, None)
        yield ('icmp'+op, 'cmp0', ty, Ty('i1', ty.elts, ty.scalable), 0, None)
        yield ('selecticmp'+op, 'cmp', ty, ty, 0, None)
        yield ('selecticmp'+op, 'cmp0', ty, ty, 0, None)
    # TODO: mla?
    # TODO: fshl+const

    # TODO: uaddo, usubo, uadde, usube?
    # TODO: umulo, smulo?
    # TODO: umulh, smulh
    # TODO: ushlsat, sshlsat
    # TODO: smulfix, umulfix
    # TODO: smulfixsat, umulfixsat
    # TODO: sdivfix, udivfix
    # TODO: sdivfixsat, udivfixsat

    for instr in ['add', 'mul', 'and', 'or', 'xor', 'smin', 'smax', 'umin', 'umax']:
      for ty in inttypes():
        if ty.elts == 1:
          continue
        yield ("reduce."+instr, 'reduce', ty, Ty(ty.scalar), 0, None)

  pool = multiprocessing.Pool(16)
  data = pool.starmap(do, enumint())
  with open(f"data-int{'-'+args.mattr if args.mattr else ''}.json", "w") as f:
    json.dump(data, f, indent=1)


if args.type == 'all' or args.type == 'fp':
  def enumfp():
    # Floating point Binops
    for instr in ['fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'minnum', 'maxnum', 'minimum', 'maximum', 'copysign', 'pow']:
      for ty in fptypes():
        for (variant, extrasize) in binop_variants(ty):
          yield (instr, variant, ty, ty, extrasize, None)

    # FP unops
    for instr in ['fneg', 'fabs', 'sqrt', 'ceil', 'floor', 'trunc', 'rint', 'nearbyint']:
      for ty in fptypes():
        yield (instr, 'unop', ty, ty, 0, None)
    for instr in ['fma', 'fmuladd']:
      for ty in fptypes():
        yield (instr, 'triop', ty, ty, 0, None)

    # FP triops
    for instr in ['fma', 'fmuladd', 'select']:
      for ty in fptypes():
        yield (instr, 'triop', ty, ty, 0, None)

    for op in ['oeq', 'ogt', 'oge', 'olt', 'ole', 'one', 'ord', 'ueq', 'ugt', 'uge', 'ult', 'ule', 'une', 'uno']:
      for ty in fptypes():
        yield ('fcmp'+op, 'cmp', ty, Ty('i1', ty.elts, ty.scalable), 0, None)
        yield ('fcmp'+op, 'cmp0', ty, Ty('i1', ty.elts, ty.scalable), 0, None)
        yield ('selectfcmp'+op, 'cmp', ty, ty, 0, None)
        yield ('selectfcmp'+op, 'cmp0', ty, ty, 0, None)

    # TODO: fmul+fadd?
    # TODO: fminimumnum, fmaximumnum
    # TODO: fpowi
    # TODO: sin, cos, etc
    # TODO: fexp, fexp2, flog, flog2, flog10
    # TODO: fldexp, frexmp

    for instr in ['fadd', 'fmul', 'fmin', 'fmax', 'fminimum', 'fmaximum']:
      for ty in fptypes():
        if ty.elts == 1:
          continue
        yield ("reduce."+instr, 'reduce', ty, Ty(ty.scalar), 0, None)

  pool = multiprocessing.Pool(16)
  data = pool.starmap(do, enumfp())
  with open(f"data-fp{'-'+args.mattr if args.mattr else ''}.json", "w") as f:
    json.dump(data, f, indent=1)


if args.type == 'all' or args.type == 'castint':
  def enumcast():
    for instr in ['zext', 'sext']:
      for ty1 in inttypes():
        for ty2 in inttypes(True):
          if ty1.elts != ty2.elts or ty1.scalable != ty2.scalable or ty1.scalarsize() >= ty2.scalarsize():
            continue
          yield (instr, 'cast '+ty2.scalar, ty1, ty2, 0, None)
    for instr in ['trunc']:
      for ty1 in inttypes(True):
        for ty2 in inttypes(True):
          if ty1.elts != ty2.elts or ty1.scalable != ty2.scalable or ty1.scalarsize() <= ty2.scalarsize():
            continue
          yield (instr, 'cast '+ty2.scalar, ty1, ty2, 0, None)

  pool = multiprocessing.Pool(16)
  data = pool.starmap(do, enumcast())
  with open(f"data-castint{'-'+args.mattr if args.mattr else ''}.json", "w") as f:
    json.dump(data, f, indent=1)


if args.type == 'all' or args.type == 'castfp':
  def enumcast():
    for instr in ['fptosi', 'fptoui']:
      for ty1 in fptypes():
        for ty2 in inttypes():
          if ty1.elts != ty2.elts or ty1.scalable != ty2.scalable:
            continue
          yield (instr, 'cast '+ty2.scalar, ty1, ty2, 0, None)
    for instr in ['sitofp', 'uitofp']:
      for ty1 in fptypes():
        for ty2 in inttypes():
          if ty1.elts != ty2.elts or ty1.scalable != ty2.scalable:
            continue
          yield (instr, 'cast '+ty2.scalar, ty2, ty1, 0, str(ty1))
    for instr in ['fpext']:
      for ty1 in fptypes():
        for ty2 in fptypes():
          if ty1.elts != ty2.elts or ty1.scalable != ty2.scalable or ty1.scalarsize() >= ty2.scalarsize():
            continue
          yield (instr, 'cast '+ty2.scalar, ty1, ty2, 0, None)
    for instr in ['fptrunc']:
      for ty1 in fptypes():
        for ty2 in fptypes():
          if ty1.elts != ty2.elts or ty1.scalable != ty2.scalable or ty1.scalarsize() <= ty2.scalarsize():
            continue
          yield (instr, 'cast '+ty2.scalar, ty1, ty2, 0, None)

    # TODO: fpext, fptrunc, fptosisat, fptouisat
    # TODO: lrint, llrint, lround, llround

  pool = multiprocessing.Pool(16)
  data = pool.starmap(do, enumcast())
  with open(f"data-castfp{'-'+args.mattr if args.mattr else ''}.json", "w") as f:
    json.dump(data, f, indent=1)


if args.type == 'all' or args.type == 'vec':
  def enumvec():
    for instr in ['insertelement', 'extractelement']:
      for ty in inttypes():
        if ty.elts == 1:
          continue
        for variant in ['vecop0', 'vecop1', 'vecopvar']:
          yield (instr, variant, ty, ty, 0, None)
      for ty in fptypes():
        if ty.elts == 1:
          continue
        for variant in ['vecop0', 'vecop1', 'vecopvar']:
          yield (instr, variant, ty, ty, 0, None)

  # TODO: shuffles

  pool = multiprocessing.Pool(16)
  data = pool.starmap(do, enumvec())
  with open(f"data-vec{'-'+args.mattr if args.mattr else ''}.json", "w") as f:
    json.dump(data, f, indent=1)
