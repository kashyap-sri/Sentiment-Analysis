$('.input-group-btn').click(function() {
    insertMessage();
});

$(window).on('keydown', function(e) {
    if (e.which == 13) {
        insertMessage();
        return false;
    }
})

function insertMessage() {

    msg = $('.form-control').val();
    console.log(msg)

    if ($.trim(msg) == '') {
        return false;
    }

    var arr = { "tag": msg };
    $.ajax({
         url: 'http://localhost:8080/message',
         data: JSON.stringify(arr),
         type: 'POST',
         dataType: "json",
         contentType: "application/json",
         success: function(response) {

            $("#p1").html(response["pos"][0])
            $("#p2").html(response["pos"][1])
            $("#p3").html(response["pos"][2])
            $("#n1").html(response["neg"][0])
            $("#n2").html(response["neg"][1])
            $("#n3").html(response["neg"][2])
            
            setTimeout(function () {
                chart.load({
                    columns: [['data', response["sentiment"]]]
                });
            }, 1000);
         },
         error: function(error) {
          console.log("error calling function")
          }
     });    
}

