import sys, os, subprocess, argparse, logging, json

# Try to more extensively check the cost model figures coming out of the cost model, for every operation x type combo.
# Currently it looks at costsize costs, as those are easier to measure.
# Measures codesize from llc with some filtering.
# Can add other costs in the future, they are more difficult to measure correctly.

# Run this to generate data.json
#   python llvm/utils/costmodeltest.py
# and this to serve is to pert 8081, inside a venv with pandas
#   python llvm/utils/costmodeltest.py --servellvm/utils/costmodeltest.py

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='')
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def run(cmd):
  logging.debug('> ' + cmd)
  cmd = cmd.split()
  return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
  
def getcost(costkind, print):
  text = run(f"opt {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} costtest.ll -passes=print<cost-model> -cost-kind={costkind} -disable-output")
  costpre = 'Cost Model: Found an estimated cost of '
  if print:
    logging.debug(text.strip())
  costs = [x for x in text.split('\n') if 'instruction:   ret ' not in x]
  cost = sum([int(x[len(costpre):len(costpre)+x[len(costpre):].find(' ')]) for x in costs if x.startswith(costpre)]) 
  return (cost, text.strip())

def checkcosts(llasm):
  logging.debug(llasm)
  with open("costtest.ll", "w") as f:
    f.write(llasm)
  
  run(f"llc {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} costtest.ll -o costtest.s")
  #run(f"llc {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} costtest.ll -o costtest.o -filetype=obj --function-sections")
  #sizelines = run(f"llvm-size -A costtest.o")
  #size = int([x for x in sizelines.split('\n') if ".text.test" in x][0].split()[1]) // 4 - 1
  with open("costtest.s") as f:
    lines = [l.strip() for l in f]
  # This tries to remove .declarations, comments and invariant instructions (movs and constants).
  lines = [l for l in lines if l[0] != '.' and l[0] != '/' and not l.startswith('test:')]
  #logging.debug(lines)
  # TODOD: Improve the filtering to what is invariant, somehow. Or include it in the costs.
  filteredlines = [l for l in lines if not l.startswith('movi') and not l.startswith('mov\tw') and l != 'ret' and not l.startswith('adrp') and not l.startswith('ldr') and not l.startswith('dup') and not l.startswith('fmov')]
  logging.debug(filteredlines)
  size = len(filteredlines)
  logging.debug(f"size = {size}")

  codesize = getcost('code-size', True)
  thru = getcost('throughput', False)
  lat = getcost('latency', False)
  sizelat = getcost('size-latency', False)

  logging.debug(f"cost = codesize:{codesize[0]} throughput:{thru[0]} lat:{lat[0]} sizelat:{sizelat[0]}")
  return (size, [codesize, thru, lat, sizelat], llasm, ('\n'.join(lines)).replace('\t', ' '))

  if args.checkopted:
    run(f"opt {'-mtriple='+args.mtriple if args.mtriple else ''} {'-mattr='+args.mattr if args.mattr else ''} costtest.ll -O1 -S -o -")


def generate_const(ty, sameval):
  consts = ['3', '2'] if not ty.isFloat() else ['3.0', '2.0']
  if ty.elts == 1:
    return consts[0]
  if sameval:
    return "<" + ", ".join([f"{ty.scalar} {consts[0]}" for x in range(ty.elts)]) + '>'
  return "<" + ", ".join([f"{ty.scalar} {consts[0]}, {ty.scalar} {consts[1]}" for x in range(ty.elts // 2)]) + '>'

def generate(variant, instr, ty):
  tystr = ty.str()
  eltstr = ty.scalar
  preamble = f"define {tystr} @test({tystr} %a"
  if variant == 'binop':
    preamble += f", {tystr} %b"
  elif variant == 'binopsplat':
    preamble += f", {eltstr} %b"
  preamble += ") {\n"

  setup = ""
  if variant == "binopsplat":
    setup += f"  %i = insertelement {tystr} poison, {eltstr} %b, i64 0\n  %bs = shufflevector {tystr} %i, {tystr} poison, <{ty.elts} x i32> zeroinitializer\n"

  instrstr = "  %c = "
  b = "%b" if variant == "binop" or variant == "binopsplat" else generate_const(ty, variant == 'binopconstsplat')
  if instr in ['add', 'sub', 'mul', 'sdiv', 'srem', 'udiv', 'urem', 'and', 'or', 'xor', 'shl', 'ashr', 'lshr', 'fadd', 'fsub', 'fmul', 'fdiv', 'frem']:
    instrstr += f"{instr} {tystr} %a, {b}\n"
  elif instr in ['rotr', 'rotl']:
    instrstr += f"call {tystr} @llvm.fsh{instr[3]}({tystr} %a, {tystr} %a, {tystr} {b})\n"
  else:
    instrstr += f"call {tystr} @llvm.{instr}({tystr} %a, {tystr} {b})\n"
  
  return preamble + setup + instrstr + f"  ret {tystr} %c\n}}"

class Ty:
  def __init__(self, scalar, elts=1):
    self.scalar = scalar
    self.elts = elts
  def isFloat(self):
    return self.scalar[0] != 'i'
  def str(self):
    if self.elts == 1:
      return self.scalar
    return f"<{self.elts} x {self.scalar}>"
  def __repr__(self):
    return self.str()
fptymap = { 16:'half', 32:'float', 64:'double',
            'half':16, 'float':32, 'double':64 }

def integertypes():
  #TODO: i128, other type sizes?
  for bits in [8, 16, 32, 64]:
    yield Ty('i'+str(bits))
  for bits in [8, 16, 32, 64]:
    for s in [2, 4, 8, 16, 32]:
      if s * bits > 256: #TODO: Higher sizes are incorrect for codesize
        continue
      if s * bits < 64: #TODO: Start looking at smaller sizes once the legal sizes are better. Odd vector sizes
        continue
      yield Ty('i'+str(bits), s)
def fptypes():
  #TODO: f128? They are just libcalls
  for bits in [16, 32, 64]:
    yield Ty(fptymap[bits])
  for bits in [16, 32, 64]:
    for s in [2, 4, 8, 16, 32]:
      if s * bits > 256: #TODO: Higher sizes are incorrect for codesize
        continue
      if s * bits < 64: #TODO: Start looking at smaller sizes once the legal sizes are better. Odd vector sizes
        continue
      yield Ty(fptymap[bits], s)

def binop_variants(ty):
  yield ('binop', 0)
  #yield ('binopconst', 0) #1 if ty.elts == 1 else 2)
  #if ty.elts > 1:
  #  yield ('binopsplat', 0) #1
  #  yield ('binopconstsplat', 0) #1 if ty.elts == 1 or ty.bits <= 32 else 2)

parser = argparse.ArgumentParser()
parser.add_argument('--serve', action='store_true')
parser.add_argument('-mtriple', default='aarch64')
parser.add_argument('-mattr', default=None)
parser.add_argument('-checkopted', action='store_true')
args = parser.parse_args()

if args.serve:
  import pandas, numpy, html
  df = pandas.read_json("data.json")
  df['error'] = df.codesize - df['size']
  df2 = pandas.pivot_table(df, index=['instr', 'variant'], columns=['ty'],
                           values=['error', 'size', 'thru', 'lat', 'sizelat', 'codesize', 'll', 'asm', 'costoutput'],
                           aggfunc=lambda x: x)
  df3 = pandas.pivot_table(df, index=['instr', 'variant'], columns=['ty'],
                           values=['error'])

  def to_html(tys):
    asm = '<table border="1" class="dataframe"><thead><tr>\n'
    asm += '  <th>instr</th>\n'
    asm += '  <th>variant</th>\n'
    for ty in tys:
      asm += f'  <th>{ty}</th>\n'
    asm += '</tr></thead>\n'

    asm += '<tbody>\n'
    for idx in df2.index:
      print(idx)
      item = df2.loc[idx]
      if pandas.isna(item['error'][tys[-1]]):
        continue

      asm += '  <tr>\n'
      asm += f'    <th rowspan="1" valign="top">{html.escape(str(idx[0]))}</th>\n'
      asm += f'    <th>{html.escape(str(idx[1]))}</th>\n'
      for ty in tys:
        #print("  ", ty)
        error = item["error"][ty]
        if pandas.isna(error):
          error = "-"
        size = item["size"][ty]
        codesize = item["codesize"][ty]
        thru = item["thru"][ty]
        lat = item["lat"][ty]
        sizelat = item["sizelat"][ty]
        ll = '  ' + item["ll"][ty].replace('\n', '\n  ')
        asmtxt = '  ' + item["asm"][ty].replace('\n', '\n  ')
        costout = '  ' + item["costoutput"][ty].replace('\n', '\n  ')
        tt = f'{idx[0]} {idx[1]} {ty}\nCodesize cost:{codesize}\nMeasured size:{size}\nOthercosts: thru:{thru} lat:{lat} sizelat:{sizelat}\nIR:\n{ll}\nAsm:\n{asmtxt}\nCodesize cost output:\n{costout}'
        tt = html.escape(tt).replace('\\n', '\n').replace('\n', '&#013;')
        asm += f'    <td title="{tt}">{html.escape(str(error))}</th>\n'
      asm += '  </tr>\n'
    asm += '</tbody></table>\n'
    return asm

  html_template = f"""
<html>
<head>
    <style>
        h1 {{
          font-family: verdana;
          font-size: 24px;
        }}
        h3 {{
          font-family: verdana;
          font-size: 20px;
        }}
        table {{
            width: 60%;
            margin: 10px auto;
            border-collapse: collapse;
            background: white;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }}
        th, td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #4c79aa;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        td:hover {{
          background-color: #ddd;
        }}
    </style>
</head>
<body>
    <h1>Cost model tests</h1>
    <p>This attempts to compare the first-order output of the cost model compared to the measured codesize. It generates simple snippets of IR and compares the output of opt -passes=print<cost-model> -cost-kind=codesize and the assmebly generated from llc. The assembly output of llc is sometimes filtered to remove loop pinvariant instructions. The table lists the difference between the two cost models.</p>
    <p>Not all the scores are expected to match exactly. More details on each and the costs found can be found in their respective tooltips.</p>
    <h3>Integer errors</h3>
    {to_html([str(x) for x in integertypes()])}
    <h3>Float errors</h3>
    {to_html([str(x) for x in fptypes()])}
    {df3.to_html()}
</body>
</html>
"""
  with open("data.html", "w", encoding="utf-8") as f:
    f.write(html_template)

  import http.server
  server = http.server.HTTPServer(('', 8081), http.server.SimpleHTTPRequestHandler)
  print("http://e108579-lin.cambridge.arm.com:8081/data.html")
  server.serve_forever()



data = []
try:
  def do(instr, variant, ty, extrasize):
    logging.info(f"{variant} {instr} with {ty.str()}")
    (size, costs, ll, asm) = checkcosts(generate(variant, instr, ty))
    if costs[0][0] != size - extrasize:
      logging.warning(f">>> {variant} {instr} with {ty.str()}  size = {size} vs cost = {costs[0][0]} (expected extrasize={extrasize})")
    logging.debug('')
    data.append({"instr":instr, "ty":str(ty), "variant":variant, "codesize":costs[0][0], "thru":costs[1][0], "lat":costs[2][0], "sizelat":costs[3][0], "size":size, "extrasize":extrasize, "asm":asm, "ll":ll, "costoutput":costs[0][1]})

  # Operations are the ones in https://github.com/llvm/llvm-project/issues/115133
  #  TODO: load/store, bitcast, getelementptr, phi, select, icmp, zext/sext/trunc, not?

  # Integer Binops
  for instr in ['add', 'sub', 'mul', 'sdiv', 'srem', 'udiv', 'urem', 'and', 'or', 'xor', 'shl', 'ashr', 'lshr', 'smin', 'smax', 'umin', 'umax', 'uadd.sat', 'usub.sat', 'sadd.sat', 'ssub.sat', 'rotr', 'rotl']:
    for ty in integertypes():
      for (variant, extrasize) in binop_variants(ty):
        do(instr, variant, ty, extrasize)

  # Integer unops
  # abs, bitreverse, bswap, ctlz, cttz, ctpop, 
  # Integer triops
  # fshl, fshr, rotr, rotl, 
  # select, icmp, fcmp

  #uaddo, usubo, uadde, usube?
  #umulo, smulo?
  #umulh, smulh
  #ushlsat, sshlsat
  #smulfix, umulfix
  #smulfixsat, umulfixsat
  #sdivfix, udivfix
  #sdivfixsat, udivfixsat


  # Floating point Binops
  for instr in ['fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'minnum', 'maxnum', 'minimum', 'maximum', 'copysign', 'pow']:
    for ty in fptypes():
      for (variant, extrasize) in binop_variants(ty):
        do(instr, variant, ty, extrasize)

  # fma, fmuladd
  # fneg, fabs, fsqrt, ceil, floor, trunc, rint, nearbyint
  # fpext, fptrunc, fptosi, fptoui, uitofp, sitofp, fptosisat, fptouisat
  # lrint, llrint, lround, llround
  # fminimumnum, fmaximumnum
  # fpowi
  # sin, cos, etc
  # fexp, fexp2, flog, flog2, flog10
  # fldexp, frexmp

except KeyboardInterrupt:
  pass
with open("data.json", "w") as f:
  json.dump(data, f)
print(data)