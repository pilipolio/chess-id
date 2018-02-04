addOverlay = function() {
    var $src = $('#grid-source');
    var $wrap = $('<div id="grid-overlay"></div>');

    var $nCells = 8;
    var $cellSize = Math.ceil($src.find('img').innerWidth() / $nCells);
    var $cellSize = $src.find('img').innerWidth() / $nCells;
    console.log($cellSize)

    // create overlay
    var $tbl = $('<table></table>');
    $tbl.css('border-spacing', '0px').css('border-collapse', 'collapse');

    for (var row = 0; row < $nCells; row++) {
        var $tr = $('<tr></tr>');
        for (var col = 0; col < $nCells; col++) {
            var $td = $('<td></td>');
            $td.data("row", row);
            $td.data("col", col);
            $td.data("piece_index", 0);
            $td.text(grid[row][col][0]);
            $td.css('color', 'red').css('align', 'center').css('width', ($cellSize-2)+'px').css('height', $cellSize+'px');
            $td.addClass('unselected');
            $tr.append($td);
        }
        $tbl.append($tr);
    }
    $src.css('width', $nCells*$cellSize+'px').css('height', $nCells*$cellSize+'px')

    // attach overlay
    $wrap.append($tbl);
    $src.after($wrap);

    $('#grid-overlay td').hover(function() {
        $(this).toggleClass('hover');
    });

    $('#grid-overlay td').click(function() {
        var row = $(this).data("row");
        var col = $(this).data("col");
        var next_piece_index = ($(this).data("piece_index") + 1) % 13;
        $(this).data("piece_index", next_piece_index);
        console.log(next_piece_index)
        $(this).text(grid[row][col][next_piece_index]);
        $(this).toggleClass('selected').toggleClass('unselected');
    });
}

exportPieces = function(imageId) {
    var pieces = $("td").map(function(index) { return grid[$(this).data("row")][$(this).data("col")][$(this).data("piece_index")];}).get();
    $.post(url='/annotations/' + imageId, data=JSON.stringify({'pieces':pieces}), dataType='json', contentType='application/json')
}