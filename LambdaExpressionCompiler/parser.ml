open LccTypes 

let match_token (toks : 'a list) (tok : 'a) : 'a list =
  match toks with
  | [] -> raise (Failure("List was empty"))
  | h::t when h = tok -> t
  | h::_ -> raise (Failure( 
      Printf.sprintf "Token passed in does not match first token in list"
    ))

let lookahead toks = match toks with
   h::t -> h
  | _ -> raise (Failure("Empty input to lookahead"))


(* Write your code below *)

let rec parse_lambda toks =
  let (body, rest) = parse_aux toks in
  if rest <> [Lambda_EOF] then
    raise (Failure "parsing failed")
  else
    body

and parse_aux toks = 
  match toks with
  | Lambda_Var s :: t ->
    (Var (s), t)
  | Lambda_LParen :: Lambda_Lambda :: Lambda_Var s :: Lambda_Dot :: t ->
    let (body, rest) = parse_aux t in
    (Func(s, body), match_token rest Lambda_RParen)
  | Lambda_LParen :: t -> 
    let (func, rest) = parse_aux t in
    let (arg, rest') = parse_aux rest in
    let rest'' = match_token rest' Lambda_RParen in
    (Application (func, arg), rest'')
  | _ -> raise (Failure "parsing failed")

let rec parse_engl toks =
  let (body, rest) = parse_c toks in
  if rest <> [Engl_EOF] then
    raise (Failure "parsing failed")
  else
    body

and parse_c toks = 
  match lookahead toks with
  | Engl_If -> 
    let rest = match_token toks Engl_If in
    let (condition, rest') = parse_c rest in
    let rest'' = match_token rest' Engl_Then in
    let (true_expr, rest''') = parse_c rest'' in
    let rest'''' = match_token rest''' Engl_Else in
    let (false_expr, rest''''') = parse_c rest'''' in
    (If (condition, true_expr, false_expr), rest''''')
  | _ -> parse_h toks 

and parse_h toks =
  let (lhs, rest) = parse_u toks in
  if rest = [] then (lhs,rest) else
  match lookahead rest with
  | Engl_And -> 
    let rest' = match_token rest Engl_And in
    let (rhs, rest'') = parse_h rest' in
    (And (lhs, rhs), rest'')
  | Engl_Or ->
    let rest' = match_token rest Engl_Or in
    let (rhs, rest'') = parse_h rest' in
    (Or (lhs, rhs), rest'')
  | _ -> (lhs, rest)

and parse_u toks =
  match lookahead toks with
  | Engl_Not -> 
    let rest = match_token toks Engl_Not in
    let (body, rest') = parse_u rest in
    (Not (body), rest')
  | _ -> parse_m toks

and parse_m toks =
  match lookahead toks with
  | Engl_True -> 
    let rest = match_token toks Engl_True in
    (Bool (true), rest)
  | Engl_False -> 
    let rest = match_token toks Engl_False in
    (Bool (false), rest)
  | Engl_LParen ->
    let rest = match_token toks Engl_LParen in
    let (body, rest') = parse_c rest in
    let rest'' = match_token rest' Engl_RParen in
    ((body), rest'')
  | _ -> raise (Failure "parsing failed")


  
