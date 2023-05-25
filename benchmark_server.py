import requests
import concurrent.futures
from time import perf_counter_ns
session = requests.session()

req = requests.Request('POST', "http://localhost:5000/search", data={"text":"#![allow(unused_imports)]\nuse std::io::{self, prelude::*};\nuse std::str;\nuse std::mem::swap;\nuse std::cmp::*;\nuse std::collections::*;\n\nfn solve<R: BufRead, W: Write>(scan: &mut Scanner<R>, w: &mut W){\n    let mut t = scan.token::<i16>();\n    // let mut t = 1;\n    while { t -= 1; t + 1 } > 0 {\n        let n = scan.token::<u64>();\n        let s = scan.token::<String>();\n        let mut mx = 0;\n        \n        for ch in s.chars() {\n            mx = max(mx, ch as i32);\n        }\n        \n        writeln!(w, \"{}\", mx - ('a' as i32) + 1);\n    }\n}\n\nfn main(){\n    let (stdin, stdout) = (io::stdin(), io::stdout());\n    let mut scan = Scanner::new(stdin.lock());\n    let mut out = io::BufWriter::new(stdout.lock());\n\n    solve(&mut scan, &mut out);\n}\n\n// Source: https://codeforces.com/profile/EbTech\n// List of codes taken: imports, Scanner struct and its implementation,\n// codes inside main function, and solve function's definition\nstruct Scanner<R> {\n    reader: R,\n    buf_str: Vec<u8>,\n    buf_iter: str::SplitWhitespace<'static>,\n}\nimpl<R: BufRead> Scanner<R> {\n    fn new(reader: R) -> Self {\n        Self { reader, buf_str: vec![], buf_iter: \"\".split_whitespace() }\n    }\n    fn token<T: str::FromStr>(&mut self) -> T {\n        loop {\n            if let Some(token) = self.buf_iter.next() {\n                return token.parse().ok().expect(\"Failed parse\");\n            }\n            self.buf_str.clear();\n            self.reader.read_until(b'\\n', &mut self.buf_str).expect(\"Failed read\");\n            self.buf_iter = unsafe {\n                let slice = str::from_utf8_unchecked(&self.buf_str);\n                std::mem::transmute(slice.split_whitespace())\n            }\n        }\n    }\n}","n_results":100,"tags":["implementation","strings","greedy"]}).prepare()

def search():
    session.send(req)

for n in [5, 10, 15, 20]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        # Start the load operations and mark each future with its URL
        for m in [100, 10000, 1000000]:
            suc_cnt, err_cnt = 0, 0
            ts = perf_counter_ns()
            future_to_url = [executor.submit(search) for _ in range(m)]
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    data = future.result()
                    suc_cnt += 1
                except Exception as exc:
                    # print(exc)
                    err_cnt += 1

            te = perf_counter_ns()          
            print(f"worker: {n}, req_cnt: {m}, suc_cnt: {suc_cnt}, err_cnt: {err_cnt}, time: {(te-ts)/10**9:.9f}")