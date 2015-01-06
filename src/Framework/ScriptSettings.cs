namespace Mpdn.RenderScript
{
    namespace Config
    {
        namespace Internal
        {
            public class RenderScripts
            {
            }
        }

        public abstract class ScriptSettings<TSettings> 
            : ScriptSettingsBase<Internal.RenderScripts, TSettings> where TSettings : class, new()
        {
            protected ScriptSettings()
            {
            }

            protected ScriptSettings(TSettings settings)
                : base(settings)
            {
            }
        }
    }
}