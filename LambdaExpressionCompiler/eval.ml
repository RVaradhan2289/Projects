open LccTypes 

let cntr = ref (-1)

let fresh () =
  cntr := !cntr + 1 ;
  !cntr

let rec lookup env var = match env with 
  [] -> None
  |(v,e)::t -> if v = var then e else lookup t var


let alpha_convert e = 
  cntr := 0 ;
  let rec convert_aux env e = 
    match e with
    | Var s -> 
      (match lookup env s with
      | Some Var x -> Var x
      | None -> Var s)
    | Func (s, e1) -> 
      let new_env = (s, Some (Var (string_of_int (fresh())))) :: env in
      let Some Var x = lookup new_env s in
      Func (x, convert_aux new_env e1)
    | Application (e1, e2) -> Application ((convert_aux env e1, convert_aux env e2))
  in convert_aux [] e

let rec isalpha e1 e2 =
  let new_e1 = alpha_convert(e1) in
  let new_e2 = alpha_convert(e2) in
  new_e1 = new_e2

let reduce env e = 
  let rec reduce_aux env e = 
    match e with
    | Var s ->
      if lookup env s = None then
        Var s
      else
        let Some x = lookup env s in
        x 
    | Func (s, e1) ->
      Func (s, reduce_aux ((s, None) :: env) e1)
    | Application (Func (s, e1'), e2) ->
      let arg = reduce_aux env e2 in
      let new_env = (s, Some arg) :: env in
      reduce_aux new_env e1'
    | Application (e1, e2) -> 
      let func = reduce_aux env e1 in
      let arg = reduce_aux env e2 in
      if func = e1 && arg = e2 then
        Application (func, arg)
      else
        reduce_aux env (Application(func, arg))
  in reduce_aux env (alpha_convert(e))

let rec laze env e = 
  let new_e = alpha_convert(e) in
  match new_e with
    | Var s ->
      if lookup env s = None then
        Var s
      else
        let Some x = lookup env s in
        x 
    | Func (s, e1) ->
      Func (s, laze ((s, None) :: env) e1)
    | Application (Func (s, e1'), e2) ->
      let new_env = (s, Some e2) :: env in 
      laze new_env e1'
    | Application (e1, e2) ->
      let arg = laze env e2 in
      Application (e1, arg)  

let rec eager env e = 
  let new_e = alpha_convert(e) in
  match e with
    | Var s ->
      if lookup env s = None then
        Var s
      else
        let Some x = lookup env s in
        x 
    | Func (s, e1) ->
      Func (s, eager ((s, None) :: env) e1)
    | Application (Func (s, e1'), e2) ->
      let arg = eager env e2 in
      if arg = e2 then
        let new_env = (s, Some e2) :: env in 
        eager new_env e1'
      else
        Application (Func (s, e1'), arg)
    | Application (e1, e2) ->
      let arg = eager env e2 in
      Application (e1, arg)  

let rec convert tree =
  match tree with
  | Bool true -> "(Lx.(Ly.x))"
  | Bool false -> "(Lx.(Ly.y))"
  | If (a,b,c) -> "((" ^ (convert a) ^ " " ^ (convert b) ^ ")" ^ " " ^ (convert c) ^ ")"
  | Not (a) -> "((Lx.((x (Lx.(Ly.y))) (Lx.(Ly.x))))" ^ " " ^ (convert a) ^ ")"
  | And (a,b) -> "(((Lx.(Ly.((x y) (Lx.(Ly.y))))) " ^ (convert a) ^ ")" ^ " " ^ (convert b) ^ ")"
  | Or (a,b) -> "(((Lx.(Ly.((x (Lx.(Ly.x))) y))) " ^ (convert a) ^ ")" ^ " " ^ (convert b) ^ ")"
  | _ -> raise (Failure "eval failed")

let readable tree =
  let new_tree = alpha_convert(tree) in
  let rec readable_aux tree = 
    match tree with
    | Func (x, Func (y, Var z)) when x = z ->
      "true"
    | Func (x, Func (y, Var z)) when y = z ->
      "false"
    | Application (Func (s, Application (Application (Var r, Func (x, Func (y, Var z))), Func (n, Func (m, Var l)))), a) 
    when s = r && y = z && n = l ->
      "(not " ^ (readable_aux a) ^ ")"
    | Application (Application (Func (s, Func (r, Application (Application (Var t, Var u), Func (x, Func (y, Var z))))), a), b)
    when s = t && r = u && y = z->
      "(" ^ (readable_aux a) ^ " and " ^ (readable_aux b) ^ ")"
    | Application (Application (Func (s, Func (r, Application (Application (Var t, Func (x, Func (y, Var z))), Var u))), a), b) 
    when s = t && x = z && r = u->
      "(" ^ (readable_aux a) ^ " or " ^ (readable_aux b) ^ ")"
    | Application (Application (a, b), c) ->
      "(if " ^ (readable_aux a) ^ " then " ^ (readable_aux b) ^ " else " ^ (readable_aux c) ^ ")"
  in
  readable_aux new_tree