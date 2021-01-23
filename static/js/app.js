var data = []
var token = ""

jQuery(document).ready(function () {
    $('#input_question').keyup(function (e) {
        if (e.which === 13) {
            $('#btn-process').click()
        }
    });

    $('#btn-process').on('click', function () {
        input_question = $('#input_idx').val()

        $.ajax({
            url: '/get_answer',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_idx": input_question
            }),
            beforeSend: function () {
                $('.overlay').show()
                $('#question').val('')
                $('#text_paragraphs').val('')
                $('#first').val('')
                $('#random').val('')
                $('#bert').val('')
                $('#bart').val('')
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#question').val(jsondata['question'])
            $('#text_paragraphs').val(jsondata['reviews'])
            $('#first').val(jsondata['first'])
            $('#random').val(jsondata['random'])
            $('#bert').val(jsondata['bert'])
            $('#bart').val(jsondata['bart'])
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            alert(jsondata['responseText'])
        });
    })


})