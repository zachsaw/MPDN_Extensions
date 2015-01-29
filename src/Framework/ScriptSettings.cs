namespace Mpdn.RenderScript
{
    namespace Config
    {
        namespace Internal
        {
            public class RenderScripts
            {
                // No implementation - we are only using this class as our folder name to pass into ScriptSettingsBase
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