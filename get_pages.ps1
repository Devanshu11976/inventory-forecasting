$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open('C:\Users\devan\Downloads\AI-Based Inventory Forecasting for Small Businesses.docx')
$pages = $doc.ComputeStatistics(2)
Write-Host "EXACT MS WORD PAGE COUNT: $pages"
$doc.Close(0)
$word.Quit()
