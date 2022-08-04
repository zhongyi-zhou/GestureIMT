function post_image(image) {
        $("#status").empty().text("File is uploading...");
        $(this).ajaxSubmit({
            error: function(xhr) {
        status('Error: ' + xhr.status);
            },

            success: function(response) {
        $("#status").empty().text(response);
                console.log(response);
            }
    });
    return false;

}

function uploadStatus(mydict, wrapper=null) {
    if (wrapper != null) 
    {let newdict = {}; newdict[wrapper] = mydict; mydict=newdict;}
    // console.log(mydict)
    return new Promise((res, rej) => {
        $.ajax({
            url: "/post_status",
            type: "POST",
            data: JSON.stringify(mydict),
            contentType: "application/json; charset=utf-8",
            processData: false,
            success: function(data) {
                // console.log('Result: Upload successful')
                res(data)
            },
            error: function(e) {
                console.log('Result: Error occurred: ' + e.message)
                rej(e)
            }
        });
    })

}


function post_test(posturl) {
    $.ajax({ 
        url: posturl, 
        type: 'POST', 
        data: JSON.stringify({
            "test": "test",
        }), 
        contentType: "application/json; charset=utf-8",
        processData: false, 
        dataType: "json",
        success: function(response){ 
            console.log(response)
            console.log("success!")
        }, 
    }); 
     
}

