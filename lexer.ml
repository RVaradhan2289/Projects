open LccTypes

let lex_lambda input =
    let rec lambda_list input pos =
        let var_re = Str.regexp "[a-z]" in
        let lParen_re = Str.regexp "(" in
        let rParen_re = Str.regexp ")" in
        let lamb_re = Str.regexp "L" in
        let dot_re = Str.regexp "\\." in
        let whitespace_re = Str.regexp "[ \t\n]+" in

        if pos < String.length input then
            if Str.string_match whitespace_re input pos then
                let len = Str.match_end () - pos in
                lambda_list input (pos + len)
            else if Str.string_match var_re input pos then
                let matched_var = Str.matched_string input in
                Lambda_Var matched_var :: lambda_list input (pos + 1)
            else if Str.string_match lamb_re input pos then
                Lambda_Lambda :: lambda_list input (pos + 1)
            else if Str.string_match dot_re input pos then
                Lambda_Dot :: lambda_list input (pos + 1)
            else if Str.string_match lParen_re input pos then
                Lambda_LParen :: lambda_list input (pos + 1)
            else if Str.string_match rParen_re input pos then
                Lambda_RParen :: lambda_list input (pos + 1)
            else
                raise (Failure "tokenizing failed")
        else
            [Lambda_EOF]
    in
    lambda_list input 0

let lex_engl input = 
    let rec engl_list input pos =
        let and_re = Str.regexp "and" in
        let or_re = Str.regexp "or" in
        let not_re = Str.regexp "not" in
        let if_re = Str.regexp "if" in
        let then_re = Str.regexp "then" in
        let else_re = Str.regexp "else" in
        let true_re = Str.regexp "true" in
        let false_re = Str.regexp "false" in
        let lParen_re = Str.regexp "(" in
        let rParen_re = Str.regexp ")" in
        let whitespace_re = Str.regexp "[ \t\n]+" in

        if pos < String.length input then
            if Str.string_match whitespace_re input pos then
                let len = Str.match_end () - pos in
                engl_list input (pos + len)
            else if Str.string_match and_re input pos then
                Engl_And :: engl_list input (pos + 3)
            else if Str.string_match or_re input pos then
                Engl_Or :: engl_list input (pos + 2)
            else if Str.string_match not_re input pos then
                Engl_Not :: engl_list input (pos + 3)
            else if Str.string_match if_re input pos then
                Engl_If :: engl_list input (pos + 2)
            else if Str.string_match then_re input pos then
                Engl_Then :: engl_list input (pos + 4)
            else if Str.string_match else_re input pos then
                Engl_Else :: engl_list input (pos + 4)
            else if Str.string_match true_re input pos then
                Engl_True :: engl_list input (pos + 4)
            else if Str.string_match false_re input pos then
                Engl_False :: engl_list input (pos + 5)
            else if Str.string_match lParen_re input pos then
                Engl_LParen :: engl_list input (pos + 1)
            else if Str.string_match rParen_re input pos then
                Engl_RParen :: engl_list input (pos + 1)
            else
                raise (Failure "tokenizing failed")
        else
            [Engl_EOF]
    in
    engl_list input 0
