-module(cholesky).
-author('evgeny').
-export([solver/2, fromFile/1]).

-type dim() :: non_neg_integer().
-type scalar() :: number().
-type vector() :: list(scalar()).
-type matrix() :: list(vector()).

fromFile(Filename) ->
  case file:open(Filename, [read]) of
    {ok, IoDevice} ->
      List = readFile(IoDevice, []),
      A = from_list(List),
      b = row(0, A),
      file:close(IoDevice),
      StartTime = erlang:system_time(millisecond),
      solver(A, b),
      EndTime = erlang:system_time(millisecond),
      io:format("Time: ~p~n", [(EndTime - StartTime)/1000]),
      file:close(IoDevice);
    {error, Reason} ->  io:format("~p", [Reason])
  end.

readFile(IoDevice, List ) ->
  case file:read_line(IoDevice) of
    {ok, Data} ->
      Res = lists:map(
        fun(String) ->
          list_to_integer(String) end,
        string:tokens(Data, " ")),
      readFile(IoDevice, lists:append(List, Res));
    eof -> List
  end.

-spec from_list([[number()]]) -> matrix().
from_list(ListOfLists) when length(ListOfLists) > 0 ->
    NumRows = length(ListOfLists),
    NumColsInFirstRow = length(hd(ListOfLists)),
    {true, _} = {lists:all(fun(Row) ->
				   length(Row) == NumColsInFirstRow
			   end, ListOfLists),
		 "Matrix must not be jagged"},
    array:from_list(lists:flatten(ListOfLists)).

-spec solver(matrix(), matrix()) -> matrix().
solver(L,b) ->
    StartTime = erlang:system_time(millisecond),
    L_T = transpose(L),
    y = solve(L_T,b),
    solve(L,y),
    EndTime = erlang:system_time(millisecond),
    io:format("Time: ~p~n", [(EndTime - StartTime)/1000]).

-spec choletsky(matrix()) -> matrix().
choletsky(RowWise) ->
    choletsky(0, RowWise).

choletsky(N, Matrix) when N == length(Matrix) ->
    Matrix;
choletsky(N, Matrix) ->
    
    {Top, Bottom} = lists:split(N, Matrix),
    {[SL | Left], [[H | Row] | Rows]} = lists:unzip([lists:split(N, Rs) || Rs <- Bottom]),
    {Col, Minor} = lists:unzip([lists:split(1, Rs) || Rs <- Rows]),
   
    S = sumsq(SL),
    NewH = sqrt(H - S),
    NewRow = [0 || _ <- Row],
    NewCol = [
        [(Y - dot(SL, lists:sublist(row(J + 1, [SL | Left]), N))) / NewH]
     || {J, [Y]} <- lists:zip(lists:seq(1, length(Col)), Col)
    ],
 
    NewBottom = [
        A ++ B ++ C
     || {A, B, C} <- lists:zip3([SL | Left], [[NewH] | NewCol], [NewRow | Minor])
    ],
    choletsky(N + 1, lists:append(Top, NewBottom)).

sqrt(X) when X < 0 ->
    erlang:error({error, matrix_not_positive_defined});
sqrt(X) ->
    math:sqrt(X).

% Reference
-spec row(dim(), matrix()) -> vector() | matrix().
row(0, _) ->
    [];
row(I, Matrix) when I > 0 ->
    lists:nth(I, Matrix);
row(I, Matrix) when I < 0 ->
    {A, [_ | B]} = lists:split(-(I + 1), Matrix),
    lists:append(A, B).

-spec col(dim(), matrix()) -> vector() | matrix().
col(0, _) ->
    [];
col(J, Matrix) when J > 0 ->
    [lists:nth(J, Row) || Row <- Matrix];
col(J, Matrix) when J < 0 ->
    Colbind = fun({A, [_ | B]}) -> lists:append(A, B) end,
    [Colbind(lists:split(-(J + 1), Row)) || Row <- Matrix].

% Transformation
-spec transpose(matrix()) -> matrix().
transpose([[]]) ->
    [];
transpose([[X]]) ->
    [[X]];
transpose([[] | Rows]) ->
    transpose(Rows);
transpose([[X | Xs] | Rows]) ->
    [[X | [H || [H | _] <- Rows]] | transpose([Xs | [Tail || [_ | Tail] <- Rows]])].

% Sum Product 
-spec dot(vector(), vector()) -> scalar().
dot(VecA, VecB) ->
    lists:foldl(fun(X, Sum) -> Sum + X end, 0, lists:zipwith(fun(X, Y) -> X * Y end, VecA, VecB)).

% Matrix Multiplication
-spec matmul(matrix(), matrix()) -> matrix().
matmul(M1 = [H1 | _], M2) when length(H1) =:= length(M2) ->
    matmul(M1, transpose(M2), []).
matmul([], _, R) ->
    lists:reverse(R);
matmul([Row | Rest], M2, R) ->
    matmul(Rest, M2, [outer(Row, M2) | R]).

% Reductions
sum(X) when is_number(X) -> X;
sum(Xs) -> reduction(Xs, fun(X, Sum) -> Sum + X end, 0).

sumsq(X) when is_number(X) -> X * X;
sumsq(Xs) -> reduction(Xs, fun(X, SumSq) -> SumSq + X * X end, 0).

prod(X) when is_number(X) -> X;
prod(Xs) -> reduction(Xs, fun(X, Acc) -> Acc * X end, 1).

% Solves
-spec solve(matrix(), matrix()) -> matrix().
solve(X, B) ->
    matmul(X, B).

% private arithmetic functions
reduction([], _Fun, Init) ->
    Init;
reduction([X | _] = Vector, Fun, Init) when is_number(X) ->
    lists:foldl(Fun, Init, Vector);
reduction([V | _] = Matrix, Fun, Init) when is_list(V) ->
    reduction(lists:flatten(Matrix), Fun, Init).
